# EEG Data Analysis Tools

Python tools for analyzing EEG data logged by the Sleep Headband firmware.

## Installation

Install required Python packages:

```bash
pip install numpy matplotlib scipy
```

## Usage

### Reading and Analyzing Logged Data

After running the batch inference test on the Teensy, you'll have several files on the SD card:

- `batch_XXXXX_raw.bin` - Raw filtered EEG samples (after bandpass filter)
- `batch_XXXXX_normalized.bin` - Normalized windows sent to the model
- `batch_XXXXX_metadata.txt` - Metadata about the data files
- `batch_test_XXXXX.csv` - Inference results (sleep stage predictions)

Copy these files from the SD card to your computer, then run:

```bash
python read_eeg_data.py batch_XXXXX
```

This will generate:

1. **Visualizations:**
   - `raw_eeg.png` - Time series plot of raw filtered data
   - `normalized_window_epoch0.png` - First normalized window
   - `window_comparison.png` - Comparison of multiple windows
   - `spectrogram.png` - Frequency analysis (0-35 Hz)

2. **CSV Exports:**
   - `raw_data_first_30s.csv` - First 30 seconds of raw data
   - `normalized_windows.csv` - First 5 normalized windows

## Comparing with Reference Model

To compare your data with your co-worker's model inputs:

1. Get the input data format from your co-worker's implementation
2. Load both datasets in Python:

```python
import numpy as np

# Your data
your_windows = read_normalized_windows("batch_XXXXX_normalized.bin")

# Reference data (adjust format as needed)
reference_data = np.loadtxt("reference_input.csv")

# Compare
window_idx = 0
your_data = your_windows[window_idx][1]
ref_data = reference_data[:3000]

print("Mean difference:", np.abs(your_data - ref_data).mean())
print("Max difference:", np.abs(your_data - ref_data).max())
print("Correlation:", np.corrcoef(your_data, ref_data)[0, 1])
```

3. Check preprocessing parameters match:
   - Bandpass filter: 0.5-35 Hz (verify cutoff frequencies)
   - Z-score normalization: mean=0, std=1 per window
   - Sample rate: 100 Hz
   - Window size: 3000 samples (30 seconds)

## File Formats

### Raw Data File (`*_raw.bin`)
- Binary file of 32-bit floats (little-endian)
- Sequential samples at 100 Hz
- After bipolar derivation and bandpass filtering
- Before z-score normalization

Read in Python:
```python
data = np.fromfile("batch_XXXXX_raw.bin", dtype=np.float32)
```

### Normalized Data File (`*_normalized.bin`)
- Binary file with structured windows
- Each window:
  - 4 bytes: epoch_index (int32)
  - 4 bytes: window_size (int32)
  - window_size * 4 bytes: samples (float32 array)

### Metadata File (`*_metadata.txt`)
- Human-readable text file
- Key-value pairs describing the data
- Processing pipeline documentation

## Generating Sleep Stage Summary

The inference logger already generates a summary. To visualize it:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Read inference results
df = pd.read_csv("batch_test_XXXXX.csv")

# Plot sleep stages over time
plt.figure(figsize=(14, 4))
plt.plot(df['time_seconds'] / 3600, df['predicted_stage'])
plt.xlabel('Time (hours)')
plt.ylabel('Sleep Stage')
plt.yticks([0, 1, 2, 3, 4], ['N3', 'N2', 'N1', 'REM', 'Wake'])
plt.title('Sleep Stages Over Time')
plt.grid(True, alpha=0.3)
plt.savefig('sleep_stages.png', dpi=150)
```

## Spectrogram Generation

The `read_eeg_data.py` script automatically generates a spectrogram showing:
- Frequency range: 0-35 Hz (delta, theta, alpha, beta bands)
- Time resolution: 4-second windows with 50% overlap
- Color scale: Power spectral density in dB

For custom spectrograms with different parameters, modify the `generate_spectrogram()` function.

## Troubleshooting

**"No module named 'scipy'"**: Install scipy with `pip install scipy`

**"File not found"**: Make sure you're running the script from the directory containing the data files

**"Incomplete window data"**: The firmware may have been interrupted. Check the last complete window index in the metadata.

**Normalization looks wrong**: Check that mean ≈ 0 and std ≈ 1. If not, there may be an issue with the EEG processor.
