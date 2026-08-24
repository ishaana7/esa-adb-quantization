# ESA-ADB Quantization Pipeline

Two files from a larger replication study of Hundman et al.'s (2018) Telemanom
LSTM anomaly detection framework, tested on the ESA Anomaly Detection
Benchmark (ESA-ADB) Mission 1 lightweight subset. This repo holds only the
code that was actually changed from the original Telemanom implementation,
not a full copy of the pipeline.

## What's here

**`errors.py`**: a modified version of Telemanom's non-parametric dynamic thresholding module. Two constants got recalibrated once it became clear they were tuned specifically for NASA's SMAP/MSL error distributions, not ESA-ADB's: the scale-check threshold (0.05 → 0.02) and the standard deviation search limit (sd_lim, 12.0 → 8.0).

**`quantize_and_benchmark.py`**: a PyTorch pipeline built from scratch that pulls trained weights out of Telemanom's saved Keras .h5 models, rebuilds the same LSTM architecture in PyTorch, applies post-training dynamic INT8 quantization, and benchmarks FP32 vs. INT8 model size, inference speed, and prediction agreement. 

## LSTM Structure

The unmodified Telemanom codebase (Hundman et al., 2018), available at
[khundman/telemanom](https://github.com/khundman/telemanom)

## Reference

Hundman, K., Constantinou, V., Laporte, C., Colwell, I., & Soderstrom, T.
(2018). Detecting Spacecraft Anomalies Using LSTMs and Nonparametric
Dynamic Thresholding. *KDD '18*.
https://doi.org/10.1145/3219819.3219845
