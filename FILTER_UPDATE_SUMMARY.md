# Filter Update Summary

**Date**: 2025-11-05
**Purpose**: Update firmware filters to match training script preprocessing exactly

---

## Problem Identified

Your validation notebook showed **87.5% agreement** between firmware preprocessing and training preprocessing. Analysis revealed:

- **Training script** uses: 5th order Butterworth (0.5-30Hz) at 250Hz sample rate
- **Original firmware** used: 6th order Beta bandpass + notch filters at 500Hz

The filter mismatch caused systematic differences in model predictions.

---

## Changes Made

### 1. Generated Stable Filter Coefficients

**Script**: `generate_training_filter_coefficients.py`

Generated a 5th order Butterworth bandpass filter (0.5-30Hz) at 250Hz with:
- ✅ All poles inside unit circle (max magnitude: 0.996220)
- ✅ Float32 precision validated (negligible precision loss)
- ✅ Impulse response decays properly
- ✅ Correct -3dB cutoff frequencies: 0.50Hz and 29.98Hz

### 2. Created Filter Implementation

**Files Created**:
- `include/TrainingBandpassFilter.h` - Filter class declaration
- `src/TrainingBandpassFilter.cpp` - Direct Form II cascaded biquads implementation

**Implementation Details**:
- 5 cascaded second-order sections (SOS)
- Direct Form II structure (most numerically stable)
- Float32 precision matching Teensy hardware
- Proper state initialization and reset functionality

### 3. Updated Preprocessing Pipeline

**Files Modified**:
- `include/PreprocessingPipeline.h`
- `src/PreprocessingPipeline.cpp`

**New Pipeline** (matching training script):
```
4000Hz → 250Hz → Butterworth (0.5-30Hz) → 100Hz
```

**Implementation Details**:
- **Stage 1**: Downsample 4000Hz → 250Hz (average every 16 samples)
- **Stage 2**: Apply Butterworth filter at 250Hz
- **Stage 3**: Resample 250Hz → 100Hz using 5:2 rational resampling with linear interpolation

### 4. Validation Tests

**Script**: `test_filter_stability.py`

All stability tests **PASSED**:
- ✅ Pole stability (all poles inside unit circle)
- ✅ Impulse response decay
- ✅ Step response acceptable
- ✅ White noise test (no divergence)
- ✅ Typical EEG signal test

---

## Files Generated/Modified

### New Files:
1. `generate_training_filter_coefficients.py` - Coefficient generation script
2. `include/TrainingBandpassFilter.h` - Filter header
3. `src/TrainingBandpassFilter.cpp` - Filter implementation
4. `test_filter_stability.py` - Validation test script
5. `FILTER_UPDATE_SUMMARY.md` - This document
6. `validate_model_inference_updated.ipynb` - Updated validation notebook

### Modified Files:
1. `include/PreprocessingPipeline.h` - Updated to use TrainingBandpassFilter
2. `src/PreprocessingPipeline.cpp` - Implemented 4000Hz→250Hz→100Hz pipeline

### Deprecated Files (no longer used):
- `include/BetaBandpassFilter.h`
- `src/BetaBandpassFilter.cpp`
- `include/NotchFilterCascade.h`
- `src/NotchFilterCascade.cpp`

---

## Next Steps

### 1. Compile and Test on Teensy

```bash
# Build the project
pio run

# Upload to Teensy 4.1
pio run --target upload

# Monitor output
pio device monitor
```

### 2. Validate Preprocessing on Hardware

Test with the same data file used in validation notebook:
- File: `data/example_datasets/eeg/SdioLogger_miklos_night_2.bin`
- Expected: Same preprocessing output as Python validation

### 3. Re-run Validation Notebook

Run `validate_model_inference_updated.ipynb` with Teensy output to verify:
- **Target**: >95% agreement between Teensy and training preprocessing
- Previous: 87.5% agreement (with old filters)
- Expected: >95% agreement (with new filters)

### 4. Compare Model Predictions

Collect predictions from Teensy and compare with validation notebook:
- Check agreement percentage
- Verify confidence levels match
- Confirm stage distributions are similar

---

## Technical Notes

### Filter Specifications

**Training Butterworth Filter:**
- Type: 5th order Butterworth bandpass
- Cutoffs: 0.5 Hz (low), 30 Hz (high)
- Sample rate: 250 Hz
- Implementation: 5 cascaded second-order sections
- Stability: Max pole magnitude 0.996220 < 1.0

**SOS Coefficients** (5 sections):
```
Section 1: [b0, b1, b2] = [0.00258, 0.00516, 0.00258]
           [a1, a2] = [-0.95968, 0.29944]

Section 2: [b0, b1, b2] = [1.0, 2.0, 1.0]
           [a1, a2] = [-1.21206, 0.65966]

Section 3: [b0, b1, b2] = [1.0, 0.0, -1.0]
           [a1, a2] = [-1.43307, 0.44022]

Section 4: [b0, b1, b2] = [1.0, -2.0, 1.0]
           [a1, a2] = [-1.97953, 0.97969]

Section 5: [b0, b1, b2] = [1.0, -2.0, 1.0]
           [a1, a2] = [-1.99230, 0.99246]
```

### Pipeline Timing

At 4000 Hz input:
- **250Hz output**: Every 16 samples (4ms)
- **100Hz output**: Every 80 samples (20ms)

This means the model receives a new 30-second epoch every 3000 samples at 100Hz (30 seconds).

### Memory Usage

**Per PreprocessingPipeline instance:**
- Downsampling buffer (250Hz): 16 floats = 64 bytes
- Resample buffer (5:2): 5 floats = 20 bytes
- Filter state (5 sections × 2): 10 floats = 40 bytes
- **Total**: ~124 bytes per instance

---

## Troubleshooting

### If agreement is still <95%:

1. **Check normalization** - Verify z-score calculation matches training
2. **Check data types** - Ensure float32 throughout pipeline
3. **Check filter state** - Verify filter is reset at epoch boundaries
4. **Check resampling** - Validate 250Hz→100Hz interpolation

### If filter appears unstable on hardware:

1. Run `test_filter_stability.py` again
2. Check for arithmetic overflow in fixed-point operations
3. Verify filter state variables initialized to zero
4. Check for NaN/Inf values in intermediate results

### If compilation fails:

1. Verify `TrainingBandpassFilter.h` is in `include/` directory
2. Verify `TrainingBandpassFilter.cpp` is in `src/` directory
3. Check PlatformIO build settings in `platformio.ini`
4. Clean and rebuild: `pio run --target clean && pio run`

---

## Expected Results

After deployment, you should see:

✅ **>95% agreement** between Teensy and training preprocessing
✅ **Similar confidence levels** across all sleep stages
✅ **Consistent stage distributions** matching training data predictions
✅ **Stable real-time operation** on Teensy 4.1

If you achieve <85% agreement, there may be remaining implementation differences to address.

---

## References

- Training script: `data/example_datasets/debug/5_training_script.py`
- Validation notebook: `validate_model_inference_updated.ipynb`
- Filter validation plots: `training_filter_validation.png` (if matplotlib installed)
- Preprocessing comparison: `preprocessing_comparison_firmware_vs_training.png`

---

## Contact

For questions about this update, refer to:
- `generate_training_filter_coefficients.py` - Detailed filter generation process
- `test_filter_stability.py` - Comprehensive stability tests
- This document - Implementation overview

**Status**: ✅ Ready for hardware testing
