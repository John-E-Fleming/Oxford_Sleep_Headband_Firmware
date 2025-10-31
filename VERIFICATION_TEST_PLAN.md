# Verification Test Plan

## Goal
Verify that the updated firmware preprocessing pipeline produces identical outputs to the colleague's old implementation, ensuring the ML model makes the same predictions on the same dataset.

## Test Data
- **Location**: `example_datasets/debug/` and `example_datasets/eeg/`
- **Old implementation**: `example_code/example_old_inference_code/comp_comp_firm 1.ino`
- **Old sample rate**: 500Hz
- **New sample rate**: 4000Hz (from ADS1299)
- **Example file**: `SdioLogger_miklos_night_2.bin` (4000Hz)

## Pipeline Comparison

### Old Implementation (500Hz)
```
500Hz raw → Beta BP filter → Notch filters → Downsample to 100Hz (avg every 5)
→ Batch Z-score normalize (3000 samples) → Quantize → CNN
```

### New Implementation (4000Hz)
```
4000Hz raw → Downsample to 500Hz (avg every 8) → Beta BP filter → Notch filters
→ Downsample to 100Hz (avg every 5) → Batch Z-score normalize (3000 samples)
→ Quantize → CNN
```

**Key Point**: After the first downsampling stage, both pipelines are **identical** (both operate at 500Hz).

## Test Stages

### Stage 1: Validate Downsampling (4000Hz → 500Hz)
**Goal**: Verify 4000Hz→500Hz downsampling produces expected results

**Test**:
1. Create synthetic 4000Hz test signal with known frequency content
2. Downsample to 500Hz using new pipeline (averaging every 8 samples)
3. Verify output is correct average and no aliasing occurs
4. Compare against MATLAB/Python reference downsampling

**Expected**: Simple averaging of every 8 samples should match reference

---

### Stage 2: Validate Beta Bandpass Filter
**Goal**: Verify beta bandpass filter coefficients match old implementation

**Test**:
1. Use colleague's 500Hz debug data as input to beta filter
2. Compare filtered output against colleague's intermediate filtered data (if available)
3. Apply impulse response test to verify filter characteristics

**Files to check**:
- Old coefficients: `comp_comp_firm 1.ino` lines 494-500
- New coefficients: `BetaBandpassFilter.h` lines 14-35

**Expected**: Identical coefficients → identical output

---

### Stage 3: Validate Notch Filters
**Goal**: Verify notch filters remove 30Hz, 50Hz, 100Hz, 150Hz

**Test**:
1. Create synthetic signal with tones at 30, 50, 100, 150 Hz
2. Apply notch filter cascade
3. Verify >20dB attenuation at notch frequencies
4. Verify passband frequencies unchanged

**Note**: Old code uses `Filters` library (`simpleNotchFIR`). New code uses IIR notch.
Behavior should be similar but may not be bit-exact.

**Files to check**:
- Old implementation: `comp_comp_firm 1.ino` lines 313-330
- New implementation: `NotchFilterCascade.cpp`

**Expected**: Similar frequency response, minimal differences in time domain

---

### Stage 4: Validate 500Hz → 100Hz Downsampling
**Goal**: Verify final downsampling matches old implementation

**Test**:
1. Compare downsampling logic (averaging every 5 samples)
2. Verify timing of sample output

**Files to check**:
- Old implementation: `comp_comp_firm 1.ino` lines 770-780
- New implementation: `PreprocessingPipeline.cpp` lines 43-56

**Expected**: Bit-exact match (simple averaging)

---

### Stage 5: Validate Batch Z-Score Normalization
**Goal**: Verify normalization produces identical outputs

**Test**:
1. Take 3000-sample window from debug dataset
2. Apply batch Z-score normalization
3. Compare mean and std calculations
4. Verify NO clipping is applied (old code didn't clip either)

**Files to check**:
- Old implementation: `comp_comp_firm 1.ino` lines 1138-1143 (`standardizeArray`)
- New implementation: `EEGProcessor.cpp` lines 110-156

**Expected**: Bit-exact match on mean/std calculation and normalized output

---

### Stage 6: End-to-End Model Prediction Test
**Goal**: Verify model produces same predictions on same input data

**Test**:
1. Use debug dataset from `example_datasets/debug/`
2. Process through complete new pipeline
3. Run inference and log predictions
4. Compare predictions against colleague's logged outputs

**What to compare**:
- CNN output values (yy0, yy1, yy2, yy3, yy4) - should match within floating point tolerance
- Predicted sleep stage (argmax of outputs)
- Confidence scores

**Files to check**:
- Old inference: `comp_comp_firm 1.ino` lines 1137-1257 (`classify_and_print_cnn`, `preprocess_buff`)
- New inference: `test_eeg_playback.cpp` + `EEGProcessor.cpp` + `MLInference.cpp`

**Expected**: Identical predictions (same stage, similar confidence)

---

## Test Implementation Steps

### Step 1: Create Python Validation Script
Create `test_ml_pipeline.py` (already exists?) to:
- Load debug data
- Implement reference preprocessing in Python
- Compare against firmware outputs
- Generate test report

### Step 2: Add Debug Output to Firmware
Update `test_eeg_playback.cpp` to output:
- Intermediate filter outputs (after each stage)
- Normalized window data (first/last 10 samples)
- CNN input tensor (first/last 10 values)
- CNN output probabilities

### Step 3: Run Side-by-Side Comparison
1. Run colleague's firmware on debug data → capture outputs
2. Run new firmware on debug data → capture outputs
3. Compare outputs stage-by-stage

### Step 4: Validate on Full Dataset
Once stages match:
1. Run full night of EEG data through new pipeline
2. Compare sleep stage predictions across entire recording
3. Calculate agreement percentage (should be ~100%)

---

## Success Criteria

### Critical (Must Pass)
- ✅ Filter coefficients match exactly
- ✅ Downsampling logic matches exactly
- ✅ Normalization matches exactly (no clipping)
- ✅ Model predictions match on debug dataset (>99% agreement)

### Important (Should Pass)
- ✅ Intermediate signal values match within 0.1% error
- ✅ CNN outputs match within 0.01 absolute difference
- ✅ Full recording predictions match >95%

### Nice to Have
- ✅ Notch filters have similar frequency response to old implementation
- ✅ Processing timing is similar or better

---

## Known Differences to Account For

1. **Notch Filter Implementation**:
   - Old: FIR notch filters from `Filters` library
   - New: IIR notch filters (custom implementation)
   - May have small numerical differences, but should be functionally equivalent

2. **Initial Filter Transients**:
   - First ~100 samples may differ due to different initial filter states
   - Should converge after transient period

3. **Floating Point Precision**:
   - Minor differences (<1e-6) due to different computation order are acceptable

---

## Test Data Requirements

From `example_datasets/`, we need:
- ✅ Raw EEG data file at 4000Hz (e.g., `SdioLogger_miklos_night_2.bin`)
- ✅ Colleague's 500Hz data (for comparison after first downsampling)
- ✅ Colleague's preprocessed data (after filtering, before normalization)
- ✅ Colleague's normalized data (after Z-score)
- ✅ Colleague's CNN output predictions

**Note**: Since your data is at 4000Hz and colleague's was at 500Hz, we need to:
1. Verify that 4000Hz→500Hz downsampling produces equivalent 500Hz signal
2. Then verify remaining pipeline matches from 500Hz onward

---

## Next Steps

1. ✅ Review this test plan with user
2. ⬜ Determine what debug data is available
3. ⬜ Set up Python validation script
4. ⬜ Add debug outputs to firmware
5. ⬜ Run Stage 6 end-to-end test first (fastest validation)
6. ⬜ If Stage 6 fails, work backwards through Stages 1-5
7. ⬜ Document any differences found and validate they're acceptable
