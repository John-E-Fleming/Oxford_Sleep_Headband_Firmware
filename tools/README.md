# EEG Data Analysis Tools

Python tools for analyzing EEG data and running sleep stage inference.

## Installation

Install required Python packages:

```bash
pip install numpy matplotlib scipy pandas scikit-learn tensorflow
```

---

## Quick Start

### Run Inference on New Data

```bash
# Basic usage with 1KHz data
python tools/run_inference.py data.bin --sample-rate 1000 --channels 12 --output predictions.csv

# With accelerometer extraction
python tools/run_inference.py data.bin --sample-rate 1000 --accel 8 9 10 --save-accel --output predictions.csv
```

### Visualize Results

```bash
# Generate hypnogram + spectrogram
python tools/visualize_session.py predictions.csv eeg_100hz.csv

# With accelerometer panel
python tools/visualize_session.py predictions.csv eeg_100hz.csv --accel accel.csv

# Time axis in hours
python tools/visualize_session.py predictions.csv eeg_100hz.csv --hours
```

### Compare Preprocessing Options

```bash
# Auto-detect and compare all validation results
python tools/compare_preprocessing_options.py

# Compare specific options
python tools/compare_preprocessing_options.py Default Option_A Option_D
```

---

## Tool Reference

### run_inference.py

Flexible Python inference script supporting configurable sample rates, channels, and accelerometer data.

**Features:**
- Supports 1KHz and 4KHz input data
- Configurable bipolar EEG channel indices
- Accelerometer data extraction (converted to g units)
- Option D preprocessing (best: 89.1% agreement)

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
# Different bipolar derivation (CH2 - CH4)
python tools/run_inference.py data.bin --bipolar-pos 2 --bipolar-neg 4 --output predictions.csv

# Using a config file
python tools/run_inference.py data.bin --config config.txt

# Save all outputs
python tools/run_inference.py data.bin --save-eeg --save-accel --accel 8 9 10 --output predictions.csv
```

---

### visualize_session.py

Generate publication-quality visualizations of sleep sessions.

**Features:**
- Hypnogram with clinical ordering (Wake at top, N3 at bottom)
- EEG spectrogram (0.5-30 Hz)
- Accelerometer panel (X/Y/Z traces or magnitude)
- Time axis in hours for long recordings
- Stage distribution summary

**Arguments:**
```
predictions            Predictions CSV file
eeg                    EEG CSV file
--dir, -d              Directory containing paired log files
--output, -o           Output image file
--probs                Also plot probability timeline
--accel                Accelerometer CSV file
--accel-mode           Display mode: 'xyz' (default) or 'magnitude'
--hours                Show time axis in hours
--no-spec              Skip spectrogram computation (faster)
```

**Examples:**
```bash
# Basic visualization
python tools/visualize_session.py predictions.csv eeg_100hz.csv

# With accelerometer magnitude
python tools/visualize_session.py predictions.csv eeg_100hz.csv --accel accel.csv --accel-mode magnitude

# Long recording (hours axis, skip spectrogram for speed)
python tools/visualize_session.py predictions.csv eeg_100hz.csv --hours --no-spec
```

---

### compare_preprocessing_options.py

Compare confusion matrices across different preprocessing pipeline options.

**Features:**
- Auto-detects all available validation results
- Side-by-side confusion matrix plots
- Per-stage recall comparison
- Pairwise agreement analysis

**Arguments:**
```
options                Specific options to compare (default: all)
--list                 List available options
--output, -o           Output filename (default: auto-generated)
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

### test_prediction_agreement.py

Test Python preprocessing against reference predictions.

**Purpose:**
- Verify preprocessing pipeline matches training code
- Compare predictions at multiple processing stages

**Usage:**
```bash
python tools/test_prediction_agreement.py
```

---

## File Formats

### Predictions CSV
```csv
epoch,timestamp_s,prob_wake,prob_n1,prob_n2,prob_n3,prob_rem,predicted_stage
0,30.0,0.918,0.059,0.020,0.000,0.000,Wake
1,60.0,0.996,0.000,0.000,0.000,0.000,Wake
```

### EEG 100Hz CSV
```csv
sample_index,timestamp_ms,eeg_uv
0,0,-12.3456
1,10,-11.8234
```

### Accelerometer CSV (g units)
```csv
sample_index,timestamp_ms,accel_x_g,accel_y_g,accel_z_g
0,0.0,0.012345,-0.998765,0.054321
1,1.0,0.012456,-0.998654,0.054432
```

**Accelerometer Conversion:** Raw values are converted to g units using: `g = raw * 16.0 / 4095.0`

### Raw Binary Data
- Sequential samples in int32, int16, or float32 format
- Channels interleaved: `[CH0_S0, CH1_S0, ..., CHn_S0, CH0_S1, CH1_S1, ...]`

---

## Preprocessing Pipeline Options

| Option | Pipeline | Agreement |
|--------|----------|-----------|
| Default | 4kHz→250Hz(decimate)→filter@250Hz→100Hz | 81.4% |
| A | 4kHz→500Hz(decimate)→100Hz(avg)→filter@100Hz | 86.4% |
| B | 4kHz→500Hz(average)→100Hz(avg)→filter@100Hz | 88.4% |
| C | 4kHz→100Hz(decimate)→filter@100Hz | 73.6% |
| **D** | **4kHz→100Hz(average 40)→filter@100Hz** | **89.1%** |

**Recommendation:** Use Option D for best accuracy.

---

## Troubleshooting

**"No module named 'tensorflow'"**: Install with `pip install tensorflow`

**"No module named 'sklearn'"**: Install with `pip install scikit-learn`

**"Model not found"**: Ensure model file exists at `data/example_datasets/debug/8_tflite_quantized_model.tflite`

**Low agreement**: Check:
- Correct sample rate specified
- Correct bipolar channel indices
- Data format matches (int32 vs float32)
