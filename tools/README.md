# Validation & Analysis Tools

Python tools for validating the sleep staging algorithm and analyzing EEG data.

## Table of Contents

- [Installation](#installation)
- [Validation Workflow](#validation-workflow)
- [Tool Reference](#tool-reference)
- [File Formats](#file-formats)
- [Preprocessing Pipeline Options](#preprocessing-pipeline-options)
- [Troubleshooting](#troubleshooting)

---

## Installation

Install required Python packages:

```bash
pip install numpy matplotlib scipy pandas scikit-learn tensorflow
```

---

## Validation Workflow

This section walks through the typical process for validating the sleep staging algorithm on new data.

### Step 1: Prepare Your Data

Ensure your EEG data file is in binary format with the correct structure:
- Sequential samples, channels interleaved
- Supported formats: int32, int16, float32

Create a `config.txt` file on the SD card:

```ini
datafile=YourDataFile.bin
sample_rate=1000          # or 4000
channels=14               # number of channels in file
format=int16              # or int32/float32
gain=24
vref=4.5
bipolar_channel_positive=0
bipolar_channel_negative=6
```

### Step 2: Run Inference

You have two options for running inference:

#### Option A: Run on Teensy (Embedded)

1. Copy your data file and `config.txt` to the SD card
2. Ensure `ENABLE_VALIDATION_MODE` is **commented out** in `platformio.ini`
3. Build and upload: `pio run --target upload`
4. Open serial monitor to watch progress
5. Predictions are saved to `/realtime_logs/<datafile>_predictions.csv`

#### Option B: Run in Python (Desktop)

```bash
# Basic usage
python tools/run_inference.py data.bin --sample-rate 1000 --channels 12 --output predictions.csv

# With all options
python tools/run_inference.py data.bin \
    --sample-rate 1000 \
    --channels 14 \
    --bipolar-pos 0 \
    --bipolar-neg 6 \
    --save-eeg \
    --output predictions.csv
```

### Step 3: Visualize Results

Generate a hypnogram visualization:

```bash
# Basic hypnogram (auto-skips spectrogram if EEG file is empty)
python tools/visualize_session.py predictions.csv eeg_100hz.csv

# With probability heatmap (shows model confidence)
python tools/visualize_session.py predictions.csv eeg_100hz.csv --show-probs --hours

# With accelerometer data
python tools/visualize_session.py predictions.csv eeg_100hz.csv --accel accel.csv --hours
```

### Step 4: Compare with Reference (Optional)

If you have ground truth labels:

```bash
# Generate confusion matrix
python tools/generate_confusion_matrix.py --teensy

# Compare preprocessing options
python tools/compare_preprocessing_options.py
```

### Step 5: Debug Pipeline Issues (If Needed)

For detailed debugging of the preprocessing pipeline:

```bash
# Compare Teensy vs Python checkpoint outputs
python tools/compare_teensy_python.py teensy_debug.txt

# Test prediction agreement
python tools/test_prediction_agreement.py
```

---

## Tool Reference

### run_inference.py

**Purpose:** Run sleep stage inference on raw EEG data using the Python implementation of the preprocessing pipeline.

**Features:**
- Supports 1kHz and 4kHz input data
- Configurable bipolar EEG channel indices
- Accelerometer data extraction (converted to g units)
- Uses Option D preprocessing (best: 89.1% agreement)

**Arguments:**
```
data_file              Input binary EEG file
--config, -c           Config file (alternative to CLI args)
--output, -o           Output predictions CSV
--sample-rate, -r      Input sample rate in Hz (default: 1000)
--channels, -n         Number of channels in file (default: 12)
--bipolar-pos          Positive electrode channel index (default: 0)
--bipolar-neg          Negative electrode channel index (default: 6)
--accel X Y Z          Accelerometer channel indices (e.g., --accel 8 9 10)
--model, -m            TFLite model path
--save-eeg             Save preprocessed 100Hz EEG to CSV
--save-accel           Save accelerometer data to CSV (in g units)
--max-epochs           Maximum number of epochs to process
```

**Examples:**
```bash
# 1kHz data with 14 channels
python tools/run_inference.py data.bin --sample-rate 1000 --channels 14 --output predictions.csv

# Different bipolar derivation (CH2 - CH4)
python tools/run_inference.py data.bin --bipolar-pos 2 --bipolar-neg 4 --output predictions.csv

# Extract accelerometer data
python tools/run_inference.py data.bin --accel 8 9 10 --save-accel --output predictions.csv

# Using a config file
python tools/run_inference.py data.bin --config config.txt
```

---

### visualize_session.py

**Purpose:** Generate publication-quality visualizations of sleep sessions.

**Features:**
- Hypnogram with clinical ordering (Wake at top, N3 at bottom)
- EEG spectrogram (0.5-30 Hz)
- **Probability heatmap** showing model confidence per epoch
- Accelerometer panel (X/Y/Z traces or magnitude)
- Time axis in hours for long recordings
- Stage distribution summary

**Arguments:**
```
predictions            Predictions CSV file
eeg                    EEG CSV file (can be empty)
--dir, -d              Directory containing paired log files
--output, -o           Output image file
--probs                Also plot probability timeline (separate figure)
--show-probs           Show probability heatmap panel instead of spectrogram
--accel                Accelerometer CSV file
--accel-mode           Display mode: 'xyz' (default) or 'magnitude'
--hours                Show time axis in hours
--no-spec              Skip spectrogram computation (faster)
```

**Examples:**
```bash
# Basic hypnogram only
python tools/visualize_session.py predictions.csv eeg_100hz.csv --no-spec

# Hypnogram + probability heatmap (recommended for analysis)
python tools/visualize_session.py predictions.csv eeg_100hz.csv --show-probs --hours

# Hypnogram + spectrogram
python tools/visualize_session.py predictions.csv eeg_100hz.csv --hours

# With accelerometer magnitude
python tools/visualize_session.py predictions.csv eeg_100hz.csv --accel accel.csv --accel-mode magnitude --hours

# Auto-find paired files in directory
python tools/visualize_session.py --dir /path/to/realtime_logs/
```

**Output:**
The visualization shows:
- **Hypnogram (top):** Color-coded sleep stages over time
- **Probability Heatmap (middle, with --show-probs):** Heat map showing probability for each stage at each epoch. Bright yellow = high probability, dark blue = low probability.
- **Spectrogram (middle, default):** EEG power spectrum over time
- **Accelerometer (bottom, optional):** Movement data

---

### generate_confusion_matrix.py

**Purpose:** Generate confusion matrices comparing embedded predictions with reference labels.

**Modes:**
1. **Python mode:** Process raw 4kHz data using Python preprocessing
2. **Teensy mode:** Load predictions from Teensy validation CSV file

**Arguments:**
```
--teensy               Use Teensy predictions from SD card
--reference            Path to reference predictions file
--output, -o           Output image file
```

**Examples:**
```bash
# Python preprocessing comparison
python tools/generate_confusion_matrix.py

# Teensy predictions comparison
python tools/generate_confusion_matrix.py --teensy
```

**Setup for Teensy mode:**
1. Copy `teensy_predictions.csv` from SD card to `data/teensy_predictions.csv`
2. Ensure reference predictions exist at `data/example_datasets/debug/3_quantized_model_predictions.npy`

---

### compare_preprocessing_options.py

**Purpose:** Compare confusion matrices across different preprocessing pipeline options (A, B, C, D, Default).

**Features:**
- Auto-detects all available validation results
- Side-by-side confusion matrix plots
- Per-stage recall comparison
- Overall agreement statistics

**Arguments:**
```
options                Specific options to compare (default: all)
--list                 List available options
--output, -o           Output filename
```

**Examples:**
```bash
# List available options
python tools/compare_preprocessing_options.py --list

# Compare all options
python tools/compare_preprocessing_options.py

# Compare specific options
python tools/compare_preprocessing_options.py Default Option_D
```

**Validation Results Location:**
Results should be placed in `data/validation_testing/<OptionName>/` with a `*_predictions.csv` file.

---

### compare_teensy_python.py

**Purpose:** Debug tool for comparing Teensy and Python preprocessing at each checkpoint in the pipeline.

**Use Case:** When Teensy and Python predictions disagree, this tool helps identify which stage of the pipeline differs.

**Checkpoints Compared:**
- **Checkpoint A:** 100Hz preprocessed signal statistics
- **Checkpoint B:** Epoch extraction boundaries
- **Checkpoint C:** Normalization statistics
- **Checkpoint D:** Model input (quantization if applicable)

**Usage:**
1. Run Teensy with `CHECKPOINT_DEBUG_EPOCHS` > 0 in `main_playback_inference.cpp`
2. Copy serial output to a file
3. Run comparison:

```bash
python tools/compare_teensy_python.py teensy_debug.txt
```

**Output:**
Shows differences at each checkpoint with acceptable thresholds.

---

### test_prediction_agreement.py

**Purpose:** Test that Python preprocessing matches the training code exactly.

**Use Case:** Verify preprocessing pipeline produces identical results to the original training environment.

**Usage:**
```bash
python tools/test_prediction_agreement.py
```

**What it Tests:**
- Bandpass filter implementation
- Downsampling method
- Z-score normalization
- Final prediction agreement with reference

---

### read_eeg_data.py

**Purpose:** Read and analyze EEG data logged by the Sleep Headband firmware during real-time recording sessions.

**Use Case:** Inspect raw logged data from firmware sessions (different from playback validation data).

**Expected Files:**
- `<session>_raw.bin` - Raw filtered samples
- `<session>_normalized.bin` - Normalized windows
- `<session>_metadata.txt` - Session metadata

**Usage:**
```bash
python tools/read_eeg_data.py batch_1234567890
```

---

### compare_filter_coefficients.py

**Purpose:** Compare filter coefficients between Python (scipy) and C++ implementations.

**Use Case:** Verify that the Butterworth bandpass filter implementation on Teensy matches the Python reference.

**Usage:**
```bash
python tools/compare_filter_coefficients.py
```

---

### compare_preprocessing.py

**Purpose:** Compare different preprocessing approaches side-by-side.

**Use Case:** Evaluate impact of different filter designs, downsampling methods, or normalization approaches.

**Usage:**
```bash
python tools/compare_preprocessing.py
```

---

## File Formats

### Predictions CSV (Output from Teensy or Python)
```csv
epoch,timestamp_s,prob_wake,prob_n1,prob_n2,prob_n3,prob_rem,predicted_stage
0,30.0,0.918,0.059,0.020,0.000,0.000,Wake
1,60.0,0.996,0.000,0.000,0.000,0.000,Wake
```

### EEG 100Hz CSV (Preprocessed signal)
```csv
sample_index,timestamp_ms,eeg_uv
0,0,-12.3456
1,10,-11.8234
```

### Accelerometer CSV (in g units)
```csv
sample_index,timestamp_ms,accel_x_g,accel_y_g,accel_z_g
0,0.0,0.012345,-0.998765,0.054321
1,1.0,0.012456,-0.998654,0.054432
```

### Raw Binary Data (Input)
- Sequential samples in int32, int16, or float32 format
- Channels interleaved: `[CH0_S0, CH1_S0, ..., CHn_S0, CH0_S1, CH1_S1, ...]`
- Sample rate typically 1000Hz or 4000Hz

### Config File (config.txt)
```ini
datafile=MyRecording.bin
sample_rate=1000
channels=14
format=int16
gain=24
vref=4.5
bipolar_channel_positive=0
bipolar_channel_negative=6

# Optional accelerometer channels
accel_channel_x=8
accel_channel_y=9
accel_channel_z=10
```

---

## Preprocessing Pipeline Options

The firmware supports multiple preprocessing pipelines. Option D is recommended.

| Option | Pipeline | Agreement | Notes |
|--------|----------|-----------|-------|
| Default | 4kHz→250Hz(decimate)→filter@250Hz→100Hz | 81.4% | Original approach |
| A | 4kHz→500Hz(decimate)→100Hz(avg)→filter@100Hz | 86.4% | Two-stage, 4kHz only |
| B | 4kHz→500Hz(average)→100Hz(avg)→filter@100Hz | 88.4% | Two-stage, 4kHz only |
| C | 4kHz→100Hz(decimate)→filter@100Hz | 73.6% | Direct, any rate |
| **D** | **input→100Hz(average)→filter@100Hz** | **89.1%** | **Best, any rate** |

**Supported Sample Rates (Option D):**
- 1000 Hz → 10:1 downsampling
- 2000 Hz → 20:1 downsampling
- 4000 Hz → 40:1 downsampling

---

## Troubleshooting

### Common Errors

**"No module named 'tensorflow'"**
```bash
pip install tensorflow
```

**"No module named 'sklearn'"**
```bash
pip install scikit-learn
```

**"Model not found"**
Ensure model file exists at expected path (check script defaults or use `--model` argument).

**"EEG data too short for spectrogram"**
The EEG 100Hz file is empty. Use `--show-probs` or `--no-spec` to skip spectrogram.

### Low Agreement Issues

If Teensy and Python predictions differ significantly:

1. **Check sample rate:** Ensure `config.txt` matches actual data
2. **Check channel indices:** Verify `bipolar_channel_positive` and `bipolar_channel_negative`
3. **Check data format:** int32 vs int16 vs float32
4. **Run checkpoint comparison:** Use `compare_teensy_python.py` to find divergence point

### Empty EEG Log File

If predictions logged but EEG file is empty:
- EEG logging may have been disabled (press 'l' to toggle)
- File may not have synced before power off
- Check serial output for "EEG logging to:" message at startup

---

## Directory Structure

```
tools/
├── README.md                      # This file
├── run_inference.py               # Main inference tool
├── visualize_session.py           # Visualization tool
├── generate_confusion_matrix.py   # Confusion matrix generation
├── compare_preprocessing_options.py  # Compare pipeline options
├── compare_teensy_python.py       # Debug checkpoint comparison
├── test_prediction_agreement.py   # Verify Python matches reference
├── read_eeg_data.py              # Read firmware logged data
├── compare_filter_coefficients.py # Compare filter implementations
└── compare_preprocessing.py       # Compare preprocessing approaches
```
