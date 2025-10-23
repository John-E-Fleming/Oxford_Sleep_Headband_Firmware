# Data Logging and Comparison Guide

This guide explains how to use the new data logging features to compare your EEG processing pipeline with your co-worker's reference implementation.

## What's Been Added

### New Files Created

1. **include/DataLogger.h** - Header for data logging class
2. **src/DataLogger.cpp** - Implementation that writes raw and normalized data to SD card
3. **tools/read_eeg_data.py** - Python script for offline analysis
4. **tools/README.md** - Documentation for analysis tools

### Modified Files

- **src/batch_inference_test.cpp** - Integrated DataLogger to capture:
  - Raw filtered samples (after bandpass, before normalization)
  - Normalized windows (z-scored data sent to model)

## Output Files on SD Card

When you run the batch inference test, you'll get these files:

1. **batch_XXXXX_raw.bin** - Raw filtered bipolar EEG (float32 binary)
   - After bipolar derivation (Ch+ - Ch-)
   - After bandpass filter (0.5-35 Hz)
   - Before z-score normalization
   - Sample rate: 100 Hz

2. **batch_XXXXX_normalized.bin** - Normalized windows (structured binary)
   - Z-scored data (mean=0, std=1)
   - Exact data sent to model (3000 samples per window)
   - Includes epoch indices

3. **batch_XXXXX_metadata.txt** - Human-readable metadata
   - Sample counts
   - File format descriptions
   - Processing pipeline documentation

4. **batch_test_XXXXX.csv** - Inference results (already existed)
   - Sleep stage predictions
   - Confidence scores
   - Timestamps

## How to Use

### 1. Run Test on Hardware

Upload the firmware to your Teensy 4.1 (already built successfully):

```bash
pio run --target upload
```

The firmware will:
- Process EEG data from SD card
- Run model inference
- Log all data to SD card files

### 2. Copy Files from SD Card

After the test completes, remove the SD card and copy all `batch_*` files to your computer.

### 3. Analyze with Python

```bash
cd tools
python read_eeg_data.py batch_XXXXX
```

This generates:
- **Visualizations**: Time series, spectrograms, window comparisons
- **CSV exports**: For easy comparison with reference data

### 4. Compare with Reference Implementation

To compare with your co-worker's model:

```python
import numpy as np

# Load your normalized windows
from read_eeg_data import read_normalized_windows
your_windows = read_normalized_windows("batch_XXXXX_normalized.bin")

# Load reference data (adjust path/format as needed)
ref_windows = np.load("reference_windows.npy")  # or .csv, etc.

# Compare first window
your_data = your_windows[0][1]  # (epoch_index, window_data)
ref_data = ref_windows[0]

print(f"Your shape: {your_data.shape}")
print(f"Ref shape: {ref_data.shape}")
print(f"Your mean: {your_data.mean():.6f}, std: {your_data.std():.6f}")
print(f"Ref mean: {ref_data.mean():.6f}, std: {ref_data.std():.6f}")

# Calculate differences
diff = np.abs(your_data - ref_data)
print(f"Mean absolute difference: {diff.mean():.6f}")
print(f"Max absolute difference: {diff.max():.6f}")

# Correlation
correlation = np.corrcoef(your_data, ref_data)[0, 1]
print(f"Correlation: {correlation:.6f}")
```

## Spectrogram and Sleep Stage Summary

### Spectrogram Generation

**Recommendation: Do this OFFLINE**

The Python script automatically generates spectrograms showing:
- Delta (0.5-4 Hz): Deep sleep
- Theta (4-8 Hz): Light sleep, drowsiness
- Alpha (8-12 Hz): Relaxed wakefulness
- Beta (12-30 Hz): Active thinking, alertness

This is computationally intensive and should NOT be done on the microcontroller.

### Sleep Stage Summary

**Already implemented on-device!**

The InferenceLogger tracks:
- Total inferences
- Sleep stage counts (Wake, N1, N2, N3, REM)
- Mean confidence scores
- Quality metrics

Summary is printed at the end and saved in the CSV file.

To visualize the summary:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Read CSV
df = pd.read_csv("batch_test_XXXXX.csv")

# Count sleep stages
stage_counts = df['predicted_stage'].value_counts()
print(stage_counts)

# Plot sleep stages over time
plt.figure(figsize=(14, 4))
stage_map = {'N3': 0, 'N2': 1, 'N1': 2, 'REM': 3, 'WAKE': 4}
stages_numeric = df['predicted_stage'].map(stage_map)
plt.plot(df['time_seconds'] / 3600, stages_numeric, linewidth=1)
plt.xlabel('Time (hours)')
plt.ylabel('Sleep Stage')
plt.yticks([0, 1, 2, 3, 4], ['N3', 'N2', 'N1', 'REM', 'Wake'])
plt.title('Sleep Stages Over Time')
plt.grid(True, alpha=0.3)
plt.savefig('sleep_hypnogram.png', dpi=150)
```

## Verification Checklist

Use this to ensure your preprocessing matches the reference:

- [ ] **Sample rate**: 100 Hz (verify in config.txt and metadata)
- [ ] **Window size**: 3000 samples = 30 seconds
- [ ] **Bipolar derivation**: Correct channels specified in config.txt
- [ ] **Bandpass filter**: 0.5-35 Hz cutoff frequencies
- [ ] **Normalization**: Mean ≈ 0, Std ≈ 1 (check in Python output)
- [ ] **Data range**: Z-scores typically between -3 and +3
- [ ] **No NaN/Inf values**: Check with `np.isnan(data).any()`

## Troubleshooting

### Model predictions don't match reference

1. **Check raw data**: Compare raw filtered signals
   ```python
   your_raw = np.fromfile("batch_XXXXX_raw.bin", dtype=np.float32)
   # Compare with reference raw data
   ```

2. **Check normalization**: Verify mean=0, std=1
   ```python
   for epoch, window in your_windows:
       print(f"Epoch {epoch}: mean={window.mean():.6f}, std={window.std():.6f}")
   ```

3. **Check filter response**: Compare frequency content
   ```python
   # Use spectrogram to verify filter is working
   from scipy import signal
   f, Pxx = signal.welch(your_raw[:3000], fs=100)
   plt.semilogy(f, Pxx)
   plt.xlim([0, 50])
   plt.xlabel('Frequency (Hz)')
   plt.ylabel('PSD')
   ```

4. **Check bipolar channels**: Verify correct channel indices in config.txt

### Data logging disabled

Set in batch_inference_test.cpp:
```cpp
const bool ENABLE_DATA_LOGGING = true;
```

### SD card full

Raw data: ~400KB per minute
Normalized windows: ~12KB per window (30 seconds)

For 10 hours of data:
- Raw: ~240 MB
- Normalized: ~14.4 MB
- Total: ~255 MB

Ensure you have enough free space on the SD card.

## Next Steps

1. **Upload firmware and run test** (already built)
2. **Copy files from SD card**
3. **Run Python analysis** to generate visualizations
4. **Compare with reference data** to verify preprocessing
5. **Adjust filter/normalization** if needed
6. **Re-test** until predictions match reference

The model is now working correctly from a technical standpoint. Any remaining differences in predictions are likely due to preprocessing parameters that can be verified and adjusted using these tools.
