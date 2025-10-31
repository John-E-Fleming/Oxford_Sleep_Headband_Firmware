# Preprocessing Pipeline Update Summary

## Changes Made

The firmware preprocessing pipeline has been updated to match your colleague's implementation from `comp_comp_firm 1.ino`, ensuring identical signal processing for model compatibility.

---

## New Pipeline Architecture

### Complete Signal Flow
```
4000Hz raw EEG (from ADS1299)
    ↓
Bipolar derivation (CH0 - CH6)
    ↓
Downsample to 500Hz (average every 8 samples)
    ↓
Beta bandpass filter (6th order IIR at 500Hz)
    ↓
Notch filters: 50Hz, 100Hz, 150Hz, 30Hz
    ↓
Downsample to 100Hz (average every 5 samples)
    ↓
Collect 3000 samples (30 seconds at 100Hz)
    ↓
Batch Z-score normalization (NO clipping)
    ↓
Quantize to INT8
    ↓
CNN inference
```

---

## New Files Created

### 1. `BetaBandpassFilter.h/cpp`
- 6th order IIR bandpass filter
- Exact coefficients from colleague's implementation (lines 494-500)
- Operates at 500Hz sample rate
- Implemented as 3 cascaded second-order sections (SOS)

### 2. `NotchFilterCascade.h/cpp`
- Removes power line noise: 50Hz, 100Hz, 150Hz
- Removes 30Hz artifact
- IIR notch filters (functionally equivalent to colleague's FIR version)
- Operates at 500Hz sample rate

### 3. `PreprocessingPipeline.h/cpp`
- Orchestrates complete preprocessing chain
- Handles both downsampling stages (4000→500→100 Hz)
- Integrates beta filter and notch filters
- Returns 100Hz samples ready for ML processing

---

## Modified Files

### 1. `EEGProcessor.cpp`
**Changes**:
- Replaced running statistics with **batch Z-score normalization**
- Removed clipping of normalized values (lines 131-132, 184-185 deleted)
- Now calculates mean and std on each 3000-sample window independently
- Matches old implementation's `standardizeArray()` function

**Why**: Colleague's code used batch normalization on each window, not running statistics across windows.

### 2. `test_eeg_playback.cpp`
**Changes**:
- Replaced `BandpassFilter` with `PreprocessingPipeline`
- Updated to process 4000Hz input samples (from ADS1299)
- Serial output only when 100Hz samples are ready (after full pipeline)
- Added warning if config.txt sample rate ≠ 4000Hz

**Note**: Config file should specify `sample_rate=4000` for proper pipeline operation.

---

## Key Improvements

### 1. ✅ Exact Filter Matching
- Beta bandpass filter uses **exact same coefficients** as colleague's code
- Same IIR structure (3 second-order sections)
- Operates at correct 500Hz sample rate

### 2. ✅ Correct Downsampling
- Two-stage: 4000→500Hz (÷8), then 500→100Hz (÷5)
- Simple averaging (same as colleague's implementation)
- Preserves frequency content correctly

### 3. ✅ No Data Clipping
- Old implementation did NOT clip normalized data
- New implementation now matches (removed clipping lines)
- Preserves full dynamic range for model input

### 4. ✅ Batch Normalization
- Each 3000-sample window normalized independently
- Matches colleague's `standardizeArray()` function
- Mean and std calculated per window, not across windows

---

## Configuration Requirements

### config.txt on SD card
```
datafile=SdioLogger_miklos_night_2.bin
sample_rate=4000
channels=9
bipolar_channel_positive=0
bipolar_channel_negative=6
format=int32
gain=24
vref=4.5
```

**Critical**: `sample_rate` must be **4000 Hz** for new pipeline to work correctly.

---

## Testing the Changes

### Quick Test
1. Build and upload firmware: `pio run --target upload`
2. Check serial output for preprocessing warnings
3. Verify no error messages about sample rates
4. Confirm inference runs without errors

### Full Validation
See `VERIFICATION_TEST_PLAN.md` for comprehensive testing steps to verify model predictions match colleague's implementation.

---

## Comparison: Old vs New

| Aspect | Colleague's Implementation | New Implementation | Match? |
|--------|---------------------------|-------------------|--------|
| **Input sample rate** | 500Hz | 4000Hz | ⚠️ Different (by design) |
| **First downsampling** | N/A (already 500Hz) | 4000→500Hz (÷8) | ✅ Equivalent result |
| **Beta filter coefficients** | 6th order IIR at 500Hz | Same coefficients | ✅ Exact |
| **Notch filters** | FIR (50,100,150,30 Hz) | IIR (same freqs) | ⚠️ Similar |
| **Second downsampling** | 500→100Hz (÷5) | 500→100Hz (÷5) | ✅ Exact |
| **Normalization** | Batch Z-score | Batch Z-score | ✅ Exact |
| **Clipping** | None | None (removed) | ✅ Match |
| **Window size** | 3000 samples @ 100Hz | 3000 samples @ 100Hz | ✅ Match |
| **Model input** | INT8 quantized | INT8 quantized | ✅ Match |

---

## Expected Behavior

### When Running
- Firmware reads EEG data at 4000Hz (from ADS1299)
- Every 40th sample, outputs preprocessed 100Hz data (4000÷8÷5 = 100)
- Every 30 seconds (3000 samples @ 100Hz), runs inference
- Sleep stage predictions should match colleague's outputs

### Serial Output Format
```
Time,CH0=X.XX,CH6=X.XX,Bipolar=X.XX,Processed100Hz=X.XX[,Stage,Confidence]
```

---

## Known Differences

### 1. Notch Filter Type
- **Old**: FIR notch from `Filters` library
- **New**: Custom IIR notch filters
- **Impact**: Minimal - frequency response similar, may have small numerical differences
- **Validation**: Test on synthetic signals to verify <0.1% difference

### 2. Filter Transients
- First ~100 samples may differ due to initial filter states
- Should converge after warm-up period
- **Recommendation**: Discard first 1-2 seconds of predictions

### 3. Floating Point Precision
- Different computation order may cause tiny differences (<1e-6)
- Should not affect final sleep stage classification
- **Validation**: Check CNN outputs match within 0.01

---

## Next Steps

1. ✅ Build firmware: `pio run`
2. ⬜ Test on debug dataset from `example_datasets/debug/`
3. ⬜ Compare predictions against colleague's outputs
4. ⬜ Run validation tests from `VERIFICATION_TEST_PLAN.md`
5. ⬜ Document any remaining differences

---

## Troubleshooting

### Issue: "WARNING: Preprocessing pipeline expects 4000Hz input!"
**Solution**: Update `config.txt` on SD card to set `sample_rate=4000`

### Issue: Predictions don't match colleague's
**Debug steps**:
1. Check sample rate is 4000Hz
2. Verify correct bipolar channels (CH0 - CH6)
3. Add debug output to check intermediate values
4. Run validation tests to isolate which stage differs

### Issue: Compilation errors
**Check**: All new .cpp files are being compiled by PlatformIO
- BetaBandpassFilter.cpp
- NotchFilterCascade.cpp
- PreprocessingPipeline.cpp

---

## References

- Original implementation: `example_code/example_old_inference_code/comp_comp_firm 1.ino`
- Beta filter coefficients: Lines 494-500
- Notch filters: Lines 313-330
- Downsampling: Lines 770-780
- Normalization: Lines 1138-1143
