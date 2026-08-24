import torch
import torch.nn as nn
import torch.quantization
import numpy as np
import pandas as pd
import h5py
import time
import os
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view
import gc

# qnnpack targets ARM/mobile, closer to actual flight hardware.
# fbgemm as fallback if it's not available.
if 'qnnpack' in torch.backends.quantized.supported_engines:
    torch.backends.quantized.engine = 'qnnpack'
else:
    torch.backends.quantized.engine = 'fbgemm'

mission_path = Path("/Users/ishaanagarwal/Downloads/ESA-Mission1")
l_s = 250  # must match Telemanom's training sequence length

# 46 was a separate training run
run_map = {
    41: '2026-07-19_09.02.16',
    42: '2026-07-19_09.02.16',
    43: '2026-07-19_09.02.16',
    44: '2026-07-19_09.02.16',
    45: '2026-07-19_09.02.16',
    46: '2026-07-22_11.10.43'
}


class TelemanoMLSTM(nn.Module):
    """
    Mirrors Telemanom's Keras architecture: two LSTM layers of 80 units,
    one dense output. Same shape means the trained Keras weights can be
    loaded directly, no retraining needed.
    """
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=80, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=80, hidden_size=80, batch_first=True)
        self.fc = nn.Linear(80, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = self.fc(out[:, -1, :])  # only the last timestep matters here
        return out


def find_lstm_weights(f, lstm_num):
    """
    Pulls one LSTM layer's weights from Telemanom's .h5 file. Layer names
    shift between Keras/TF versions, so this matches on weight shape
    instead of hardcoded paths, then sorts by key to keep lstm1/lstm2 in
    order.
    """
    lstm_name = 'lstm' if lstm_num == 1 else 'lstm_1'
    all_weights = {}

    def visitor(name, obj):
        if hasattr(obj, 'shape') and 'optimizer' not in name:
            all_weights[name] = obj[()]

    f.visititems(visitor)

    # filters down to just the LSTM kernels and biases
    lstm_candidates = {k: v for k, v in all_weights.items()
                      if len(v.shape) >= 1 and (
                          (len(v.shape) == 2 and v.shape[1] == 320) or
                          (len(v.shape) == 1 and v.shape[0] == 320)
                      )}
    kernels = {k: v for k, v in lstm_candidates.items()
               if 'recurrent' not in k and 'kernel' in k}
    rec_kernels = {k: v for k, v in lstm_candidates.items()
                   if 'recurrent_kernel' in k}
    biases = {k: v for k, v in all_weights.items()
              if 'bias' in k and v.shape == (320,) and 'optimizer' not in k}

    # dict order isn't guaranteed, sort by key to keep lstm1/lstm2 straight
    sorted_kernels = sorted(kernels.items())
    sorted_recs = sorted(rec_kernels.items())
    sorted_biases = sorted(biases.items())
    idx = 0 if lstm_num == 1 else 1
    if len(sorted_kernels) > idx and len(sorted_recs) > idx and len(sorted_biases) > idx:
        return sorted_kernels[idx][1], sorted_recs[idx][1], sorted_biases[idx][1]
    raise KeyError(f"could not find weights for {lstm_name}")


def find_dense_weights(f):
    """
    Same idea as find_lstm_weights for the dense layer. Tries common
    Keras/TF naming patterns first, falls back to a broader search if none
    hit.
    """
    patterns = [
        'model_weights/dense/sequential/dense',
        'model_weights/dense/sequential_1/dense',
        'model_weights/dense/dense',
    ]
    for pattern in patterns:
        try:
            return f[f'{pattern}/kernel'][()], f[f'{pattern}/bias'][()]
        except KeyError:
            continue

    found = {}

    def visitor(name, obj):
        if hasattr(obj, 'shape') and 'dense' in name and 'optimizer' not in name:
            found[name] = obj[()]

    f.visititems(visitor)
    kernels = [v for k, v in found.items() if 'kernel' in k]
    biases = [v for k, v in found.items() if 'bias' in k]
    if kernels and biases:
        return kernels[0], biases[0]
    raise KeyError("could not find dense weights")


def convert_lstm_weights(kernel, rec_kernel, bias, pytorch_lstm):
    """
    Keras and PyTorch store LSTM weights differently. Keras orients
    weights one way, PyTorch expects the opposite, hence the transpose.
    Bias also differs: Keras uses one vector per gate, PyTorch splits it
    into two that get summed internally, so the full bias goes into one
    and the other gets zeroed to match. Gate order (input, forget, cell,
    output) is the same in both, so no reordering needed.
    """
    W_i, W_f, W_c, W_o = np.split(kernel, 4, axis=1)
    U_i, U_f, U_c, U_o = np.split(rec_kernel, 4, axis=1)
    b_i, b_f, b_c, b_o = np.split(bias, 4)

    weight_ih = np.concatenate([W_i, W_f, W_c, W_o], axis=1).T
    weight_hh = np.concatenate([U_i, U_f, U_c, U_o], axis=1).T
    bias_vals = np.concatenate([b_i, b_f, b_c, b_o])

    pytorch_lstm.weight_ih_l0 = nn.Parameter(torch.tensor(weight_ih, dtype=torch.float32))
    pytorch_lstm.weight_hh_l0 = nn.Parameter(torch.tensor(weight_hh, dtype=torch.float32))
    pytorch_lstm.bias_ih_l0   = nn.Parameter(torch.tensor(bias_vals, dtype=torch.float32))
    pytorch_lstm.bias_hh_l0   = nn.Parameter(torch.zeros(4 * 80, dtype=torch.float32))


def build_and_quantize(h5_path, run_id, ch):
    """
    Rebuilds one channel's model in PyTorch and produces an FP32 and a
    dynamically quantized INT8 version. Quantization converts weights
    once, ahead of time. Activations get quantized separately, at
    runtime, per forward pass.
    """
    with h5py.File(h5_path, 'r') as f:
        l1k, l1r, l1b = find_lstm_weights(f, 1)
        l2k, l2r, l2b = find_lstm_weights(f, 2)
        dk, db = find_dense_weights(f)

    m = TelemanoMLSTM()
    convert_lstm_weights(l1k, l1r, l1b, m.lstm1)
    convert_lstm_weights(l2k, l2r, l2b, m.lstm2)
    m.fc.weight = nn.Parameter(torch.tensor(dk.T, dtype=torch.float32))
    m.fc.bias   = nn.Parameter(torch.tensor(db, dtype=torch.float32))
    m.eval()

    m_int8 = torch.quantization.quantize_dynamic(
        m, {nn.LSTM, nn.Linear}, dtype=torch.qint8
    )
    return m, m_int8


def load_scaled_test_values(ch):
    """
    Loads the exact test array Telemanom produced in Phase 1 instead of
    re-deriving scaling from the raw channel pickle. Phase 1 min-max
    normalized each channel to [-1, 1] using the full channel's min/max
    before splitting into train/test, so reusing data/test/<ch>.npy
    guarantees the same distribution the model was trained and tested on.
    Recomputing scaling here would put inputs on a scale the model never
    saw.
    """
    test_path = Path('data') / 'test' / f'{ch}.npy'
    test_arr = np.load(test_path)
    # telemetry value is the first feature per Telemanom's convention
    return test_arr[:, 0].astype(np.float32)


results = []
for ch_num, run_id in run_map.items():
    ch = f'channel_{ch_num}'
    h5_path = f'data/{run_id}/models/{ch}.h5'
    print(f"\n--- {ch} ---")
    try:
        model_fp32, model_int8 = build_and_quantize(h5_path, run_id, ch)
        print("models ready")

        # state_dict() keeps the packed int8 tensors, so size here
        # reflects real savings, not unpacked fp32 weights
        torch.save(model_fp32.state_dict(), f'{ch}_fp32.pt')
        fp32_size = os.path.getsize(f'{ch}_fp32.pt')
        torch.save(model_int8.state_dict(), f'{ch}_int8.pt')
        int8_size = os.path.getsize(f'{ch}_int8.pt')
        reduction = round(fp32_size / int8_size, 1)
        print(f"fp32: {fp32_size/1024:.1f} KB  →  int8: {int8_size/1024:.1f} KB  ({reduction}x smaller)")

        scaled = load_scaled_test_values(ch)

        # zero-copy windows instead of duplicating every slice,
        # matters at 7M+ points per channel
        all_windows = sliding_window_view(scaled, l_s)

        total_possible = len(scaled) - l_s
        n_windows = int(total_possible * 0.10)

        # seed derived per channel so a single channel can be rerun
        # later and still draw the same sample
        np.random.seed(1000 + ch_num)
        idx = np.random.choice(total_possible, size=n_windows, replace=False)
        idx.sort()
        print(f"running on {n_windows:,} windows ({n_windows/total_possible*100:.1f}% of test set)")

        chunk_size = 10000
        fp32_preds = []
        int8_preds = []

        # fp32 pass
        t0 = time.perf_counter()
        for start in range(0, len(idx), chunk_size):
            chunk_idx = idx[start:start+chunk_size]
            chunk = all_windows[chunk_idx]
            # view is read-only, .copy() gives torch a writable buffer
            chunk_tensor = torch.from_numpy(chunk.copy()).unsqueeze(-1)
            with torch.no_grad():
                preds = model_fp32(chunk_tensor).numpy()
            fp32_preds.append(preds)
            del chunk, chunk_tensor, preds
            if (start // chunk_size) % 10 == 0:
                print(f"  fp32: {start:,} / {len(idx):,} done")
        fp32_time = (time.perf_counter() - t0) * 1000
        fp32_preds = np.concatenate(fp32_preds).flatten()
        np.save(f'{ch}_fp32_preds.npy', fp32_preds)
        print(f"fp32 done: {fp32_time/1000/60:.1f} mins")
        del model_fp32
        gc.collect()

        # int8 pass, same windows
        t0 = time.perf_counter()
        for start in range(0, len(idx), chunk_size):
            chunk_idx = idx[start:start+chunk_size]
            chunk = all_windows[chunk_idx]
            chunk_tensor = torch.from_numpy(chunk.copy()).unsqueeze(-1)
            with torch.no_grad():
                preds = model_int8(chunk_tensor).numpy()
            int8_preds.append(preds)
            del chunk, chunk_tensor, preds
            if (start // chunk_size) % 10 == 0:
                print(f"  int8: {start:,} / {len(idx):,} done")
        int8_time = (time.perf_counter() - t0) * 1000
        int8_preds = np.concatenate(int8_preds).flatten()
        np.save(f'{ch}_int8_preds.npy', int8_preds)
        print(f"int8 done: {int8_time/1000/60:.1f} mins")

        del scaled, all_windows, model_int8, idx
        gc.collect()

        # mae between fp32 and int8 predictions, not accuracy against real telemetry
        mae = np.mean(np.abs(fp32_preds - int8_preds))
        speedup = round(fp32_time / int8_time, 1)
        print(f"speedup: {speedup}x  |  mae: {mae:.8f}")
        del fp32_preds, int8_preds
        gc.collect()

        results.append({
            'channel': ch,
            'n_windows': n_windows,
            'pct_coverage': round(n_windows/total_possible*100, 2),
            'fp32_kb': round(fp32_size/1024, 1),
            'int8_kb': round(int8_size/1024, 1),
            'reduction': reduction,
            'fp32_mins': round(fp32_time/1000/60, 2),
            'int8_mins': round(int8_time/1000/60, 2),
            'speedup': speedup,
            'pred_mae': round(mae, 8)
        })

    except Exception as e:
        # skip a bad channel instead of killing the whole run
        print(f"error: {e}")
        import traceback
        traceback.print_exc()
        continue

# save results to csv
results_df = pd.DataFrame(results)
results_df.to_csv('quantization_results.csv', index=False)

print("\n--- results ---")
print(f"{'channel':<14}{'windows':<12}{'coverage':<10}{'fp32_kb':<10}{'int8_kb':<10}{'reduction':<12}{'fp32_mins':<12}{'int8_mins':<12}{'speedup':<10}{'pred_mae'}")
for r in results:
    print(f"{r['channel']:<14}{r['n_windows']:<12}{str(r['pct_coverage'])+'%':<10}{r['fp32_kb']:<10}{r['int8_kb']:<10}{str(r['reduction'])+'x':<12}{r['fp32_mins']:<12}{r['int8_mins']:<12}{str(r['speedup'])+'x':<10}{r['pred_mae']}")
print("\nsaved to quantization_results.csv")
