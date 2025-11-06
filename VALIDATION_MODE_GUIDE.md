# Validation Mode Guide

**Purpose**: Compare Teensy firmware predictions against Python reference predictions in real-time

**Date Added**: 2025-11-05

## ⚠️ IMPORTANT: TensorFlow Lite Library Version

**CRITICAL**: There is a pending fix for the TensorFlow Lite library version. The firmware currently uses the wrong library, which will cause lower-than-expected validation agreement.

**See `PENDING_TFLITE_FIX.md` for details and resolution steps.**

Expected agreement with correct library: **94.1%**
Expected agreement with current (wrong) library: **Lower (needs testing)**

Complete the TFLite library fix before relying on validation results.

---

## Overview

Validation mode allows you to verify that your Teensy firmware implementation produces the same model predictions as your Python validation notebook. It loads reference predictions from a CSV file on the SD card and compares each epoch's prediction in real-time during the 8-hour test.

**Your Current Validation Results**:
- Python notebook agreement: **94.1%** (firmware vs training preprocessing)
- This validation mode will confirm if Teensy hardware achieves the same 94.1% agreement

---

## Quick Start

### Step 1: Prepare Reference Predictions

You've already generated this file! It should be at:
```
data/reference_predictions.csv
```

The file format is:
```csv
Epoch,Predicted_Stage,Wake,N1,N2,N3,REM
0,Wake,0.91796875,0.05859375,0.01953125,0.0,0.0
1,Wake,0.97265625,0.0078125,0.0,0.0,0.01953125
...
```

### Step 2: Copy to SD Card

Copy `reference_predictions.csv` to your SD card in the `data/` folder:
```
SD Card:
└── data/
    ├── SdioLogger_miklos_night_2.bin
    ├── config.txt
    └── reference_predictions.csv  ← Add this file
```

### Step 3: Enable Validation Mode

Edit `platformio.ini` and uncomment line 29:
```ini
build_flags =
    ...
    -DENABLE_VALIDATION_MODE  ; ← Remove the semicolon and space at the start
```

### Step 4: Compile and Upload

```bash
pio run --target clean   # Clean old build
pio run                  # Compile with validation enabled
pio run --target upload  # Upload to Teensy
```

### Step 5: Monitor Results

```bash
pio device monitor
```

You'll see output like:
```
=== VALIDATION MODE ENABLED ===
Validation ready with 960 reference predictions
===============================

Epoch 10 | Agreement: 9/10 (90.0%) | Mean MSE: 0.0023
MISMATCH Epoch 23: Teensy=N1, Reference=N2 | MSE=0.003214
Epoch 20 | Agreement: 19/20 (95.0%) | Mean MSE: 0.0021
...
========================================
VALIDATION SUMMARY
========================================
Total epochs compared: 960
Exact stage matches: 903/960 (94.1%)
Mean probability MSE: 0.002145
Mismatches: 57 (logged to serial above)
========================================
```

---

## What Gets Validated

### 1. **Exact Stage Match**
- Compares predicted stage (Wake, N1, N2, N3, REM)
- Tracks agreement percentage
- Logs all mismatches

### 2. **Probability Distribution**
- Calculates Mean Squared Error (MSE) between probability vectors
- Lower MSE = closer match
- Expected MSE: ~0.002 for good agreement

### 3. **Real-Time Reporting**
- Summary every 10 epochs
- Immediate mismatch notifications
- Final statistics at end of test

---

## Interpreting Results

### Excellent (>95% agreement)
✅ Firmware matches Python validation perfectly
✅ Ready for production deployment
✅ Filters and preprocessing working correctly

### Good (90-95% agreement)
✅ Firmware working well
⚠️ Small differences may be from:
- Resampling method (linear interpolation vs scipy)
- Floating-point precision
- Edge effects
✅ Acceptable for deployment

### Needs Investigation (<90% agreement)
❌ Significant differences detected
⚠️ Check:
- Filter coefficients match
- Preprocessing pipeline implementation
- Normalization calculations
- Model loading

---

## Disabling Validation Mode

For normal operation (without validation overhead):

1. Edit `platformio.ini`
2. Comment out line 29:
   ```ini
   ; -DENABLE_VALIDATION_MODE  ; ← Add semicolon and space at start
   ```
3. Recompile and upload

**Note**: With validation disabled, the code has zero performance overhead.

---

## Files Created

### New Files:
1. **`include/ValidationReader.h`** - Validation class header
2. **`src/ValidationReader.cpp`** - CSV parsing and comparison logic

### Modified Files:
1. **`src/test_eeg_playback.cpp`** - Added validation hooks (in `#ifdef` blocks)
2. **`platformio.ini`** - Added ENABLE_VALIDATION_MODE flag (commented by default)

---

## Memory Usage

**With validation enabled**:
- Reference predictions: ~50 bytes/epoch × 960 epochs = ~48 KB
- Teensy 4.1 has 1024 KB RAM, so this is <5% usage
- No performance impact in FAST_PLAYBACK mode

**With validation disabled**:
- Zero overhead (code not compiled in)

---

## Serial Output Format

### Periodic Updates (Every 10 Epochs):
```
Epoch 100 | Agreement: 94/100 (94.0%) | Mean MSE: 0.0019
```

### Mismatch Notifications:
```
MISMATCH Epoch 23: Teensy=N1, Reference=N2 | MSE=0.003214
```

### Final Summary:
```
========================================
VALIDATION SUMMARY
========================================
Total epochs compared: 960
Exact stage matches: 903/960 (94.1%)
Mean probability MSE: 0.002145
Mismatches: 57 (logged to serial above)
========================================
```

---

## Troubleshooting

### "Failed to open reference predictions file"
- Check file exists: `data/reference_predictions.csv` on SD card
- Check SD card is properly initialized
- Verify file path in code matches your SD structure

### "Failed to parse line"
- Check CSV format matches expected format
- Verify no extra commas or missing fields
- Check file encoding (should be UTF-8)

### Agreement much lower than expected (e.g., <80%)
1. Verify reference predictions match your test file
2. Check filter update was successful (TrainingBandpassFilter)
3. Verify preprocessing pipeline changes were applied
4. Compare probability distributions (MSE) not just stages

### Validation slowing down test
- Validation adds ~5-10ms per epoch (negligible)
- If concerned, disable serial printing: `enable_serial_plot = false`
- Or increase reporting interval (edit line 401 in test_eeg_playback.cpp)

---

## Technical Details

### CSV Parsing
- Uses `SdFile::fgets()` for line-by-line reading
- `strtok()` for CSV field parsing
- Loads entire file at startup for fast epoch lookup

### Comparison Metrics

**Stage Match**:
```cpp
bool match = (strcmp(teensy_stage, ref_stage) == 0);
```

**Probability MSE**:
```cpp
float mse = 0;
for (int i = 0; i < 5; i++) {
    float diff = teensy_probs[i] - ref_probs[i];
    mse += diff * diff;
}
mse /= 5.0;  // Mean
```

### Validation Logic Flow

```
1. Setup: Load reference_predictions.csv
2. After each inference:
   a. Look up reference for current epoch
   b. Compare stage (exact match)
   c. Calculate MSE (probability difference)
   d. Log if mismatch
   e. Update statistics
3. End of test: Print final summary
```

---

## Expected Timeline

For 8-hour test (960 epochs at 30 seconds each):
- Validation adds: ~10 seconds total overhead
- Most time is in model inference itself
- Validation is negligible compared to data processing

---

## Next Steps After Validation

### If Agreement >95%:
✅ Filter update successful!
✅ Firmware matches Python validation
✅ Ready to test with live EEG acquisition
✅ Deploy to production hardware

### If Agreement 90-95%:
✅ Good agreement achieved
📊 Document any systematic differences
✅ Consider acceptable for deployment
📝 Note resampling method differences in documentation

### If Agreement <90%:
❌ Investigate implementation differences
🔍 Compare probability distributions epoch-by-epoch
🐛 Debug preprocessing pipeline
📞 Review filter coefficients

---

## Comparison with Previous Results

| Method | Agreement | Notes |
|--------|-----------|-------|
| Old firmware (Beta+Notch) | 87.5% | Before filter update |
| **Updated firmware (TrainingBandpassFilter)** | **94.1%** | Python validation |
| **Teensy hardware validation** | **?%** | Run this test to find out! |

**Goal**: Teensy should match the 94.1% from Python validation

---

## Support

If you encounter issues:
1. Check serial output for error messages
2. Verify SD card files are correct
3. Review FILTER_UPDATE_SUMMARY.md for implementation details
4. Ensure all filter updates were applied correctly

---

## Summary

Validation mode is a powerful tool to verify your Teensy firmware produces the same predictions as your Python reference implementation. It's designed to be:

- **Easy to enable/disable**: One line in platformio.ini
- **Zero overhead when disabled**: Conditional compilation
- **Comprehensive reporting**: Real-time mismatches + final summary
- **Memory efficient**: <5% RAM usage on Teensy 4.1

Use it for your first test run to confirm everything works correctly, then disable for production operation.

**You're expected to see ~94.1% agreement matching your Python validation! Good luck! 🎯**
