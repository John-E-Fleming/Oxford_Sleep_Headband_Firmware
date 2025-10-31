# ML Pipeline Debugging Guide

This guide explains how to debug the ML inference pipeline using reference data from a working implementation.

## Overview

The debugging approach validates each step of the pipeline:
1. **Normalization** - Per-epoch z-score normalization of EEG data
2. **Quantization** - Converting float32 to int8 for TFLite inference
3. **Inference** - Running the model and comparing outputs

Two test tools are provided:
- **Python script** (`test_ml_pipeline.py`) - For quick PC-based validation
- **Firmware test** (`debug_ml_pipeline.cpp`) - For on-device debugging

## Reference Data

All reference files are in `data/example_datasets/debug/`:

| File | Description | Shape |
|------|-------------|-------|
| `1_bandpassed_eeg_single_channel.npy` | Raw EEG @ 100Hz, filtered 0.5-30Hz | (2,880,000,) = 8 hours |
| `2_standardized_epochs.npy` | Z-score normalized 30s epochs | (960, 3000) |
| `3_quantized_model_predictions.npy` | Model predictions (class indices) | (960,) |
| `4_quantized_model_probabilities.npy` | Model output probabilities | (960, 5) |
| `8_tflite_quantized_model.tflite` | TFLite model file | - |

**Sleep stage mapping:**
- 0 = N3 (Deep Sleep)
- 1 = N2 (Light Sleep)
- 2 = N1 (Very Light Sleep)
- 3 = REM Sleep
- 4 = Wake

## Method 1: PC-based Testing (Python)

### Prerequisites

```bash
pip install numpy
pip install tensorflow  # Optional, for inference testing
```

### Running the Test

```bash
python test_ml_pipeline.py
```

### What It Does

1. Loads all reference files from `data/example_datasets/debug/`
2. Tests normalization on first 10 epochs:
   - Extracts raw 30-second epoch from bandpassed EEG
   - Computes per-epoch mean and std
   - Applies z-score normalization: `(x - mean) / std`
   - Compares with reference normalized data
3. Tests inference on first 10 epochs:
   - Loads reference normalized epoch
   - Quantizes to int8 using model's scale/zero_point
   - Runs TFLite inference
   - Dequantizes output
   - Compares predictions and probabilities with reference

### Expected Output

```
============================================================
ML Pipeline Testing Script
============================================================

============================================================
Loading Reference Data
============================================================
✓ Loaded eeg: ...
✓ Loaded normalized: ...
✓ Loaded predictions: ...
✓ Loaded probabilities: ...
✓ Found model: ...

============================================================
Testing Normalization
============================================================

----------------------------------------------------------------------
Epoch 0 (Time: 0s - 30s)
----------------------------------------------------------------------
Raw epoch statistics:
  Mean: 0.123456
  Std:  1.234567
...
✓ PASS: Normalization matches reference

----------------------------------------------------------------------
Normalization Test Summary
----------------------------------------------------------------------
Epochs tested: 10
Max difference (overall):  0.00000012
Mean difference (overall): 0.00000003
✓ ALL TESTS PASSED: Normalization implementation is correct!

============================================================
Testing ML Inference
============================================================
...
✓ ALL PREDICTIONS MATCH!
```

### Interpreting Results

**Normalization Test:**
- Mean difference < 1e-4: PASS
- Mean difference > 1e-4: FAIL - Check z-score calculation

**Inference Test:**
- All predictions match: PASS
- Some predictions differ: Check quantization, model loading, or inference code
- Large probability differences (>0.1): Likely quantization issue

## Method 2: On-Device Testing (Teensy Firmware)

### Setup

1. **Prepare SD Card:**
   ```
   SD_CARD_ROOT/
   └── debug/
       ├── 1_bandpassed_eeg_single_channel.npy
       ├── 2_standardized_epochs.npy
       ├── 3_quantized_model_predictions.npy
       ├── 4_quantized_model_probabilities.npy
       └── 8_tflite_quantized_model.tflite
   ```

2. **Update platformio.ini:**

   Add the debug test to the build:
   ```ini
   build_src_filter = +<*> -<main.cpp> -<test_eeg_playback.cpp> -<main_ml.cpp> -<simple_test.cpp>
   ```

   Or create a new test environment:
   ```ini
   [env:debug_ml]
   platform = teensy
   board = teensy41
   framework = arduino
   build_src_filter = +<debug_ml_pipeline.cpp> -<main.cpp> -<*.cpp>
   ```

3. **Build and Upload:**
   ```bash
   pio run -e debug_ml --target upload
   pio device monitor
   ```

### Using the Firmware Test

Available commands (send via Serial Monitor):

| Command | Action |
|---------|--------|
| `t` | Run full pipeline test (10 epochs) |
| `n` | Test normalization only (epoch 0) |
| `i` | Test inference only (epoch 0) |
| `s` | Show statistics |

### Example Session

```
===========================================
ML Pipeline Debugging Tool
===========================================
Initializing SD card... OK
Initializing EEG processor... OK
Initializing ML inference... OK

Ready! Available commands:
  't' - Run full pipeline test (10 epochs)
  'n' - Test normalization only (epoch 0)
  'i' - Test inference only (epoch 0)
  's' - Show statistics
===========================================

> t

============================================================
Running Full Pipeline Test
============================================================

------------------------------------------------------------
Testing Epoch 0 (Time: 0s - 30s)
------------------------------------------------------------

=== Testing Normalization ===
✓ Loaded raw and reference data
Raw epoch - Mean: 0.123456, Std: 1.234567
✓ Normalized epoch using per-epoch z-score

--- Comparison Results ---
Max difference: 0.000001 at index 1234
Mean difference: 0.000000
✓ PASS: Normalization matches reference!

=== Testing Inference ===
✓ Loaded reference data
✓ Inference complete

--- Reference Output ---
Predicted class: 4 (Wake)
Probabilities:
  N3_Deep: 0.012345
  N2_Light: 0.023456
  N1_VeryLight: 0.034567
  REM: 0.045678
  Wake: 0.883954

--- Our Output ---
Predicted class: 4 (Wake)
Probabilities:
  N3_Deep: 0.012346
  N2_Light: 0.023457
  N1_VeryLight: 0.034568
  REM: 0.045679
  Wake: 0.883950

--- Comparison ---
Prediction match: ✓ YES
Max probability difference: 0.000004
Mean probability difference: 0.000002

...

============================================================
Test Summary
============================================================
Epochs tested: 10
Predictions matched: 10 / 10 (100%)
Max probability difference: 0.000012
Mean probability difference: 0.000004
Mean normalization error: 0.000001
```

## Troubleshooting

### Issue: Normalization Differs from Reference

**Symptoms:**
- Mean difference > 0.001 in normalization test
- Normalized values don't match reference

**Possible Causes:**
1. **Using running statistics instead of per-epoch statistics**
   - Reference uses per-epoch z-score: `(x - epoch_mean) / epoch_std`
   - Firmware might be using running mean/std across all samples

2. **Incorrect statistics calculation**
   - Check mean calculation in EEGProcessor.cpp
   - Check std calculation (should be population std, not sample std)

**Fix:**
The reference implementation normalizes each 30-second epoch independently:
```cpp
// Correct: Per-epoch normalization
float epoch_mean = calculate_mean(epoch_data, 3000);
float epoch_std = calculate_std(epoch_data, 3000);
for (int i = 0; i < 3000; i++) {
    normalized[i] = (epoch_data[i] - epoch_mean) / epoch_std;
}
```

### Issue: Predictions Don't Match

**Symptoms:**
- Predictions differ from reference
- Large probability differences (>0.1)

**Possible Causes:**
1. **Quantization error**
   - Check input scale and zero_point match model
   - Verify quantization formula: `int8 = round(float / scale) + zero_point`

2. **Model mismatch**
   - Ensure using correct model file
   - Check model version matches reference

3. **Input data preparation**
   - Epoch index not quantized correctly
   - Input buffer not properly filled

**Debug Steps:**
1. Print first few quantized values, compare with Python test
2. Print input tensor scale/zero_point from model
3. Print output tensor scale/zero_point from model
4. Verify epoch index is being added correctly

### Issue: Memory Issues on Device

**Symptoms:**
- Crashes during test
- "Failed to allocate buffers" error

**Solutions:**
1. Reduce `TEST_NUM_EPOCHS` in debug_ml_pipeline.cpp
2. Use external RAM for large buffers
3. Test one epoch at a time using `n` and `i` commands

## Key Implementation Notes

### Per-Epoch vs Running Statistics

**Reference implementation uses per-epoch normalization:**
- Each 30-second epoch is normalized independently
- Mean and std calculated from only that epoch's 3000 samples
- This is critical for correct model performance

**Firmware may use running statistics:**
- Running mean/std updated continuously
- This differs from reference and may cause poor predictions
- Check EEGProcessor::addFilteredSample() and getProcessedWindow()

### Correct Normalization Flow

1. Collect 3000 samples (30 seconds @ 100Hz)
2. Calculate mean of those 3000 samples
3. Calculate std of those 3000 samples
4. Normalize: `z[i] = (x[i] - mean) / std`
5. Quantize normalized values for model input

### Model Input Format

The model expects:
- **Input shape:** (1, 3001)
- **Input[0:3000]:** 3000 normalized EEG samples (int8)
- **Input[3000]:** Epoch index (int8), used for positional encoding
- **Quantization:** Applied to both EEG samples and epoch index

## Success Criteria

Your implementation is working correctly if:

1. ✓ Normalization mean difference < 1e-4
2. ✓ All predictions match reference (100%)
3. ✓ Mean probability difference < 0.01
4. ✓ Max probability difference < 0.1

## Next Steps After Debugging

Once both PC and device tests pass:

1. Integrate correct normalization into main firmware
2. Update EEGProcessor to use per-epoch statistics if needed
3. Test with live data from ADS1299
4. Validate real-time performance (inference time < 5s interval)
5. Test with extended recordings to verify stability

## Questions?

If tests fail and you can't identify the issue:

1. Run Python test first to establish baseline
2. Compare Python output with device output
3. Check specific sample values at mismatches
4. Verify .npy file reading is correct (128-byte header)
5. Review info.txt for data format details
