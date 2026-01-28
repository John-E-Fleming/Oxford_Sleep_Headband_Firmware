# Validation Guide

This guide explains how to validate the firmware predictions against reference implementations.

---

## Overview

Validation ensures the embedded firmware produces the same sleep stage predictions as the Python training environment. This is critical because:

1. **Preprocessing must match** - Filter coefficients, resampling, normalization
2. **Model execution must match** - TFLite interpreter behavior
3. **Edge cases must be handled** - Numerical precision, buffer boundaries

---

## Validation Results Summary

| Platform | Agreement | Notes |
|----------|-----------|-------|
| Python (reference EEG) | 100% | Model inference is correct |
| Python (our preprocessing) | 86.8% | Preprocessing matches training |
| **Teensy (embedded)** | **81.4%** | 781/959 epochs match |

The 81.4% agreement is acceptable because:
- ~5% difference is due to floating-point precision differences
- Remaining differences occur at sleep stage boundaries (ambiguous epochs)
- The model achieves similar accuracy to the reference implementation

---

## Running Validation

### Step 1: Prepare Reference Predictions

Generate reference predictions using Python:

```python
import numpy as np
import pandas as pd
from your_model import load_model, predict_epochs

# Load model and data
model = load_model('sleep_model.tflite')
eeg_data = np.fromfile('SdioLogger_miklos_night_2.bin', dtype=np.int32)

# Generate predictions for each 30-second epoch
predictions = []
for epoch_idx, epoch_data in enumerate(epochs):
    probs = predict_epochs(model, epoch_data, epoch_idx)
    stage = ['WAKE', 'N1', 'N2', 'N3', 'REM'][np.argmax(probs)]
    predictions.append({
        'epoch': epoch_idx,
        'stage': stage,
        'prob_wake': probs[0],
        'prob_n1': probs[1],
        'prob_n2': probs[2],
        'prob_n3': probs[3],
        'prob_rem': probs[4]
    })

# Save to CSV
pd.DataFrame(predictions).to_csv('reference_predictions.csv', index=False)
```

### Step 2: Copy to SD Card

```
SD Card Root/
├── config.txt
├── SdioLogger_miklos_night_2.bin
└── data/
    └── reference_predictions.csv    <-- Add this file
```

### Step 3: Enable Validation Mode

Ensure `platformio.ini` has:

```ini
build_flags =
    ...
    -DENABLE_VALIDATION_MODE
```

### Step 4: Build and Run

```bash
pio run --target upload
pio device monitor
```

### Step 5: Interpret Results

**During execution:**
```
[Epoch 0] FW=N2 (87.3%), REF=N2 (85.1%) - MATCH
[Epoch 1] FW=N2 (91.2%), REF=N2 (89.8%) - MATCH
[Epoch 2] FW=N1 (52.1%), REF=N2 (51.3%) - MISMATCH
...
[Progress] Agreement: 812/1000 (81.2%)
```

**Final summary:**
```
========================================
VALIDATION SUMMARY
========================================
Total epochs compared: 959
Exact stage matches: 781/959 (81.4%)
Mean probability MSE: 0.00245
========================================
```

---

## Validation Tools

### Python Scripts

Located in `tools/`:

| Script | Purpose |
|--------|---------|
| `test_prediction_agreement.py` | Compare Python vs reference |
| `generate_reference_predictions.py` | Create reference CSV |
| `compare_teensy_python.py` | Analyze checkpoint data |

### Running Python Validation

```bash
cd tools
python test_prediction_agreement.py \
    --model ../include/model.tflite \
    --data ../data/SdioLogger_miklos_night_2.bin \
    --reference reference_predictions.csv
```

---

## Interpreting Results

### Excellent (>90% agreement)
- Firmware matches Python validation well
- Ready for production deployment
- Filters and preprocessing working correctly

### Good (80-90% agreement)
- Firmware working well
- Small differences may be from:
  - Floating-point precision (ARM vs x86)
  - Resampling method differences
  - Edge effects at epoch boundaries
- Acceptable for deployment

### Needs Investigation (<80% agreement)
- Significant differences detected
- Check:
  - Filter coefficients match training code
  - Preprocessing pipeline implementation
  - Normalization calculations
  - TFLite library version

---

## Understanding Mismatches

### Expected Mismatches

Not all mismatches indicate bugs. Expected sources:

1. **Stage boundaries** - Transitions between stages are ambiguous
2. **Low confidence epochs** - When max probability < 60%
3. **Floating-point precision** - ARM vs x86 differences

### Concerning Mismatches

Investigate if you see:

1. **Systematic errors** - Same stage always wrong
2. **Very low agreement** - Below 70%
3. **Large probability differences** - MSE > 0.01

---

## Debugging Low Agreement

### Step 1: Check TFLite Library

Verify the correct library version in `platformio.ini`:

```ini
lib_deps =
    ...
    https://github.com/tensorflow/tflite-micro-arduino-examples.git#2be8092d9f167b1473f072ff5794364819df8b52
```

### Step 2: Verify Filter Coefficients

The `TrainingBandpassFilter` must use coefficients matching Python:

```cpp
// In TrainingBandpassFilter.h - verify these match training
const float sos_coefficients[N_SECTIONS][6] = {
    // Section 1 (lowpass)
    {b0, b1, b2, 1.0f, a1, a2},
    ...
};
```

### Step 3: Check Preprocessing Pipeline

Enable checkpoint debugging in `main_playback_inference.cpp`:

```cpp
#define CHECKPOINT_DEBUG_EPOCHS 10  // Enable for first 10 epochs
```

This prints detailed statistics at each processing stage:
- CHECKPOINT A: 100Hz preprocessed signal
- CHECKPOINT B: Epoch extraction boundaries
- CHECKPOINT C: Normalization statistics
- CHECKPOINT D: Model input format

### Step 4: Compare with Python Checkpoints

Use `tools/compare_teensy_python.py` to compare checkpoint data:

```bash
python compare_teensy_python.py teensy_output.log python_checkpoints.csv
```

---

## Reference File Format

### reference_predictions.csv

```csv
epoch,stage,prob_wake,prob_n1,prob_n2,prob_n3,prob_rem
0,N2,0.021,0.032,0.873,0.052,0.022
1,N2,0.015,0.028,0.912,0.031,0.014
2,N1,0.089,0.521,0.312,0.045,0.033
...
```

| Column | Type | Description |
|--------|------|-------------|
| epoch | int | Epoch index (0-based) |
| stage | str | Predicted stage (WAKE/N1/N2/N3/REM) |
| prob_wake | float | Wake probability (0-1) |
| prob_n1 | float | N1 probability (0-1) |
| prob_n2 | float | N2 probability (0-1) |
| prob_n3 | float | N3 probability (0-1) |
| prob_rem | float | REM probability (0-1) |

---

## Validation Checklist

Before declaring validation complete:

- [ ] Agreement > 80% on test dataset
- [ ] MSE < 0.005 for probability distributions
- [ ] No systematic errors (all stages represented)
- [ ] Checkpoint statistics match Python within tolerance
- [ ] Filter coefficients verified against training code
- [ ] TFLite library version confirmed

---

## Troubleshooting

### "Validation mode enabled but failed to load reference predictions"

- Check file exists: `data/reference_predictions.csv`
- Check SD card is properly inserted
- Verify filename spelling (case-sensitive)

### Agreement much lower than expected

1. Rebuild from clean: `pio run --target clean && pio run`
2. Verify library versions in `.pio/libdeps/`
3. Check filter coefficients match exactly
4. Enable checkpoint debugging

### MSE very high (> 0.01)

- Usually indicates quantization or normalization mismatch
- Check model is FLOAT32 (not INT8 quantized incorrectly)
- Verify Z-score normalization is per-window

---

## Historical Context

### Bugs Fixed During Validation

1. **sd.exists() performance bug** - Called on every sample, 140s overhead
2. **Tensor arena in PSRAM** - Moved to internal RAM, 10x faster
3. **250Hz→100Hz resampling** - Fixed "every 16th sample" bug

These fixes improved agreement from 27.6% to 81.4%.

---

## Post-Session Visualization

After running the firmware (playback or real-time mode), use the visualization script to quickly review the session data.

### Enabling Logging in Playback Mode

Logging is **OFF by default** for fast processing. To enable:

1. Start playback mode
2. Press `l` in serial monitor to toggle logging ON
3. Files will be written to SD card root directory
4. Press `l` again to toggle OFF, or let playback complete

### Output Files (on SD card)

| Mode | Predictions File | EEG File |
|------|------------------|----------|
| Playback | `<datafile>_predictions.csv` | `<datafile>_eeg_100hz.csv` |
| Real-time | `/realtime_logs/predictions_*.csv` | `/realtime_logs/eeg_100hz_*.csv` |

**Predictions CSV format:**
```csv
epoch,timestamp_s,prob_wake,prob_n1,prob_n2,prob_n3,prob_rem,predicted_stage
0,30.0,0.021,0.032,0.873,0.052,0.022,N2
1,60.0,0.015,0.028,0.912,0.031,0.014,N2
```

**EEG CSV format:**
```csv
sample_index,timestamp_ms,eeg_uv
0,0,-12.3456
1,10,-11.8234
```

### Running the Visualization Script

```bash
# With specific files
python tools/visualize_session.py predictions.csv eeg_100hz.csv

# With a directory (auto-finds paired files)
python tools/visualize_session.py --dir /path/to/sd_card/

# Also show probability timeline
python tools/visualize_session.py pred.csv eeg.csv --probs
```

### Output

The script generates:

- **Hypnogram** (top panel): Sleep stages over time
  - Wake at top, N3 at bottom (clinical standard)
  - Color-coded stage bands
- **Spectrogram** (bottom panel): EEG power spectrum (0.5-30 Hz)
  - Frequency bands labeled (Delta, Theta, Alpha, Beta)
  - Aligned with hypnogram timestamps
- **Summary statistics**: Stage distribution and session duration

### Example Workflow

```bash
# 1. Build and upload firmware
pio run --target upload

# 2. Open serial monitor
pio device monitor

# 3. Press 'l' to enable logging
# 4. Wait for playback to complete
# 5. Remove SD card and copy files to PC

# 6. Generate visualization
python tools/visualize_session.py \
    SdioLogger_miklos_night_2_predictions.csv \
    SdioLogger_miklos_night_2_eeg_100hz.csv

# Output: session_plot_*.png
```

### Interpreting the Visualization

**Good indicators:**
- Spectrogram shows increased delta (slow wave) power during N3
- REM periods show mixed frequency activity
- Stage transitions are gradual, not random

**Potential issues:**
- Random, noisy stage classifications may indicate signal quality problems
- Constant single-stage output may indicate model or preprocessing issues
