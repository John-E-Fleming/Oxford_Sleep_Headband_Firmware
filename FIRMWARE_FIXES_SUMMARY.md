# Firmware ML Pipeline Fixes - Summary Report

**Date:** 2025-10-28
**Status:** Analysis Complete, Awaiting On-Device Verification

## Test Results Summary

### Python Test (Reference Implementation)
- **Model tested:** `data/example_datasets/debug/8_tflite_quantized_model.tflite` (55KB)
- **Model type:** FLOAT32 (despite "quantized" in filename)
- **Normalization:** ✓ PERFECT MATCH (per-epoch z-score)
- **Inference accuracy:** 80.7% (775/960 epochs)
- **Model input format:** `[1, 1, 3000, 1]` shape, FLOAT32 dtype

### Per-Class Performance (Reference)
| Sleep Stage | Accuracy | Notes |
|------------|----------|-------|
| N3_Deep | 100.0% | Perfect |
| REM | 100.0% | Perfect |
| N1_VeryLight | 37.2% | Difficult stage (expected) |
| N2_Light | 18.2% | Poor (small sample size: 11 total) |
| Wake | 0.0% | Failed (confused with N3_Deep) |

### Most Common Errors
1. N1_VeryLight → N3_Deep: 126 times
2. Wake → N3_Deep: 32 times
3. N1_VeryLight → REM: 16 times

---

## Critical Issues Found in Firmware

### 1. Normalization Method ❌ INCORRECT

**Current Implementation (EEGProcessor.cpp:69-88):**
```cpp
// Uses RUNNING statistics across all samples
filtered_mean_ = old_mean + learning_rate * (sample - old_mean);
filtered_std_ = filtered_std_ + learning_rate * (abs(deviation) - filtered_std_);
```

**Required Implementation:**
```cpp
// Per-epoch z-score normalization
// For each 3000-sample window:
float epoch_mean = calculate_mean(epoch_data, 3000);
float epoch_std = calculate_std(epoch_data, 3000);
for (int i = 0; i < 3000; i++) {
    normalized[i] = (epoch_data[i] - epoch_mean) / epoch_std;
}
// NO CLIPPING!
```

**Impact:** Running statistics differ significantly from per-epoch statistics, causing incorrect normalization that doesn't match what the model was trained on.

---

### 2. Data Clipping ❌ INCORRECT

**Current Implementation (EEGProcessor.cpp:131-132):**
```cpp
// Clamp to reasonable range to prevent extreme values
if (output_buffer[i] > 5.0f) output_buffer[i] = 5.0f;
if (output_buffer[i] < -5.0f) output_buffer[i] = -5.0f;
```

**Analysis:**
- Reference data range: **-16.26 to +16.67**
- 23.5% of epochs (226/960) have values exceeding ±5.0
- Clipping distorts signal and changes model input distribution
- Example: Epoch 0 has max value of 16.67, clipped to 5.0 (70% reduction!)

**Required:** Remove clipping entirely - model was trained on full z-score range.

**Impact:** High - clipping affects nearly 1/4 of all epochs and removes important signal features.

---

### 3. Model Input Type ⚠️ NEEDS VERIFICATION

**Reference Model:**
- Type: FLOAT32 (not quantized INT8)
- Size: 55KB
- Input: `input_tensor_->data.f[i]` (float)
- Output: `output_tensor_->data.f[i]` (float)

**Firmware Model:**
- File: `src/model.cpp`
- Size: ~243KB (4.4x larger than reference!)
- Type: **UNKNOWN** - needs verification

**Current Firmware Code (MLInference.cpp:186-194):**
```cpp
// Assumes INT8 quantized model
for (int i = 0; i < MODEL_EEG_SAMPLES; i++) {
    int8_t x_quantized = input_data[i] / input_tensor_->params.scale + input_tensor_->params.zero_point;
    input_tensor_->data.int8[i] = x_quantized;
}
```

**⚠️ IMPORTANT:** Before changing this code, we MUST verify the firmware model type!

**Verification Steps:**
1. Upload `debug_ml_pipeline.cpp` to Teensy
2. Copy reference files to SD card `/debug/` folder
3. Run on-device test to check:
   - `input_tensor_->type` (should print FLOAT32 or INT8)
   - `input_tensor_->params.scale` (0.0 = no quantization)
   - Actual predictions vs reference

**If firmware model is FLOAT32 (like reference):**
- Remove quantization code
- Use `input_tensor_->data.f[i] = input_data[i]` directly
- Use `output_tensor_->data.f[i]` for output

**If firmware model is truly INT8 quantized:**
- Keep quantization code
- Verify scale and zero_point values match training

---

## Recommended Action Plan

### Phase 1: Verification (Do This First!) ✅

1. **Prepare SD Card:**
   ```
   SD_CARD/
   └── debug/
       ├── 1_bandpassed_eeg_single_channel.npy
       ├── 2_standardized_epochs.npy
       ├── 3_quantized_model_predictions.npy
       ├── 4_quantized_model_probabilities.npy
       └── 8_tflite_quantized_model.tflite
   ```

2. **Build and Upload Debug Test:**
   ```bash
   # Update platformio.ini to build debug_ml_pipeline.cpp
   pio run --target upload
   pio device monitor
   ```

3. **Run On-Device Tests:**
   - Send `t` command for full test
   - Check model input/output types
   - Compare predictions with reference (target: ~80% accuracy)

### Phase 2: Fix Normalization ✅

**Priority: HIGH** (Confirmed incorrect)

1. Modify `EEGProcessor` to support per-epoch normalization
2. Option A: Calculate stats when window is ready (before inference)
3. Option B: Store raw samples, normalize full window at inference time

**Implementation:**
```cpp
bool EEGProcessor::getProcessedWindowInt8(int8_t* output_buffer, float scale, int32_t zero_point, int epoch_index) {
    // Get raw window
    float window[MODEL_EEG_SAMPLES];
    for (int i = 0; i < MODEL_EEG_SAMPLES; i++) {
        window[i] = filtered_buffer_[start_index + i];
    }

    // Calculate per-epoch statistics
    float epoch_mean = 0.0f;
    for (int i = 0; i < MODEL_EEG_SAMPLES; i++) {
        epoch_mean += window[i];
    }
    epoch_mean /= MODEL_EEG_SAMPLES;

    float epoch_std = 0.0f;
    for (int i = 0; i < MODEL_EEG_SAMPLES; i++) {
        float diff = window[i] - epoch_mean;
        epoch_std += diff * diff;
    }
    epoch_std = sqrt(epoch_std / MODEL_EEG_SAMPLES);

    // Normalize (no clipping!)
    for (int i = 0; i < MODEL_EEG_SAMPLES; i++) {
        float normalized = (window[i] - epoch_mean) / epoch_std;
        // Quantize if needed, or pass directly as float
        output_buffer[i] = quantize(normalized, scale, zero_point);
    }

    return true;
}
```

### Phase 3: Remove Clipping ✅

**Priority: HIGH** (Confirmed incorrect)

**Files to modify:**
- `EEGProcessor.cpp:131-132` - Remove clamping
- `EEGProcessor.cpp:185-186` - Remove clamping from preprocessData()

### Phase 4: Fix Model Input (If Needed) ⚠️

**Priority: MEDIUM** (Depends on Phase 1 verification)

**Only if firmware model is FLOAT32:**
```cpp
// Replace quantization loop in MLInference.cpp:186-194
for (int i = 0; i < MODEL_EEG_SAMPLES; i++) {
    input_tensor_->data.f[i] = input_data[i];  // Direct float assignment
}
input_tensor_->data.f[MODEL_EEG_SAMPLES] = (float)epoch_index;

// Replace dequantization in MLInference.cpp:204-207
for (int i = 0; i < MODEL_OUTPUT_SIZE; i++) {
    output_data[i] = output_tensor_->data.f[i];  // Direct float read
}
```

---

## Success Criteria

After implementing fixes, the firmware should achieve:

✅ **Overall accuracy:** ~80-81% (775-780 matches / 960 epochs)
✅ **N3_Deep accuracy:** ~100%
✅ **REM accuracy:** ~100%
✅ **N1 accuracy:** ~35-40% (inherently difficult)
✅ **Mean probability difference:** <0.08
✅ **Normalization matches reference:** Mean diff <0.0001

**If you achieve significantly different results (e.g., <70% or >85%), investigate:**
- Model file differences
- Input data format
- Preprocessing pipeline differences

---

## Files Generated

All test outputs saved to `debug_outputs/`:
- `sleep_stage_comparison.png` - Hypnogram and probability heatmaps
- `probability_differences.png` - Difference heatmap showing mismatches
- `confusion_matrix.npy` - Per-class error analysis
- `mismatch_indices.npy` - List of epochs that don't match

---

## Notes

1. **Model Size Discrepancy:**
   - Reference model: 55KB
   - Firmware model: 243KB
   - Possible reasons:
     - Different model architectures
     - Reference model is optimized/compressed
     - Firmware has additional metadata/debug info
   - **Must verify on-device before assuming model format!**

2. **Quantization Naming Confusion:**
   - File named "quantized_model.tflite" but actually FLOAT32
   - Always verify tensor types programmatically
   - Check `scale == 0.0` to detect FLOAT32

3. **N1 Sleep Stage:**
   - Low accuracy (37%) is expected - this stage is transitional
   - Often confused with N3 or REM
   - Even human scorers have inter-rater variability on N1

4. **Wake State:**
   - 0% accuracy is concerning (all confused with N3)
   - May indicate training data imbalance (only 34 wake epochs)
   - Or preprocessing differences during wake periods

---

## Questions to Answer (Phase 1)

1. ❓ What type is the firmware model? (INT8 or FLOAT32)
2. ❓ What are the actual tensor scales and zero_points?
3. ❓ Why is firmware model 4.4x larger than reference?
4. ❓ Does on-device inference match reference predictions (~80%)?
5. ❓ Does on-device inference show same error patterns (N1→N3)?

**DO NOT modify quantization code until these are answered!**

---

## Contact

For questions about this analysis, refer to:
- `test_ml_pipeline.py` - Python validation script
- `debug_ml_pipeline.cpp` - On-device validation firmware
- `DEBUG_ML_PIPELINE.md` - Detailed testing guide
- `PYTHON_TESTING_SETUP.md` - Environment setup
