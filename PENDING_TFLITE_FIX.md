# Preprocessing Pipeline Validation Status

**Last Updated**: 2026-01-28

---

## Current Status: Ready for Teensy Validation

### Summary

Our normalization and TFLite inference code is **correct** (100% agreement when using reference preprocessed data). However, our preprocessing pipeline from raw 4kHz data produces **83% agreement** with reference predictions due to differences in how the data was originally preprocessed for model training.

**Best Method Found:** "Every 16th sample" downsampling achieves **86.8% agreement** - the highest of all methods tested (including 500Hz intermediate sampling).

### Next Step
Run validation on Teensy hardware to confirm Python results translate to firmware performance. Expected: ~86-87% agreement with "every 16th sample" method, or ~83% with current averaging method.

---

## Phase 1 Results: Python Prediction Agreement

### Test 1: Using Reference Preprocessed EEG
```bash
python tools/test_prediction_agreement.py
```
**Result**: **100% agreement** (960/960 epochs match)

This confirms:
- Z-score normalization is correct
- TFLite inference code is correct
- Model loading and FLOAT32 input handling is correct

### Test 2: Using Raw 4kHz Data with Our Preprocessing
**Result**: **83% agreement** (799/960 epochs match)

---

## Root Cause Analysis

### Data Source Investigation

| File | Correlation with Reference | Notes |
|------|---------------------------|-------|
| 4kHz SdioLogger file | **0.99** | Reference was created from this |
| 250Hz file | 0.31 | Different processing, NOT the source |

### Preprocessing Pipeline Comparison

**Our current method:**
```
4kHz raw data
    ↓ (average every 16 samples)
250Hz downsampled
    ↓ (Butterworth bandpass 0.5-30Hz)
250Hz filtered
    ↓ (scipy.signal.resample)
100Hz output
```

**Reference method (unknown exact implementation):**
```
4kHz raw data
    ↓ (unknown downsampling - possibly with anti-aliasing filter)
250Hz downsampled
    ↓ (Butterworth bandpass 0.5-30Hz)
250Hz filtered
    ↓ (scipy.signal.resample)
100Hz output
```

### Per-Class Accuracy Breakdown

| Sleep Stage | Reference Count | Our Correct | Accuracy |
|-------------|-----------------|-------------|----------|
| Wake        | 413             | 274         | **66.3%** |
| N1          | 11              | 8           | 72.7%    |
| N2          | 226             | 219         | **96.9%** |
| N3          | 276             | 274         | **99.3%** |
| REM         | 34              | 24          | 70.6%    |

### Confusion Matrix (Reference → Our Prediction)

```
        Wake    N1      N2      N3      REM
Wake     274     10      98       7      24
N1         0      8       1       0       2
N2         0      0     219       7       0
N3         0      0       2     274       0
REM        0      1       8       1      24
```

### Key Insight

**Most errors are Wake → N2 (98 epochs, 61% of all errors)**

The subtle differences in downsampling create small signal variations that primarily affect Wake/light sleep discrimination. Deep sleep detection (N2/N3) remains highly accurate (97-99%).

---

## Alternative Downsampling Methods Tested

### Methods Tested

| Method | Correlation | Agreement | Notes |
|--------|-------------|-----------|-------|
| Simple averaging (current) | 0.9898 | **83.2%** | Current Teensy implementation |
| scipy.signal.decimate (FIR) | 0.9868 | 82.4% | Worse |
| scipy.signal.decimate (IIR) | 0.9894 | 83.0% | Similar |
| Filter at 4kHz first | nan | 43.0% | Unstable |
| Zero-phase filter (filtfilt) | -0.19 | 78.0% | Much worse |
| Lower filter order (3) | 0.5729 | 82.4% | Worse |
| **Every 16th sample** | 0.9889 | **86.8%** | **Best result!** |

### Best Result: Every 16th Sample (Simple Decimation)

Taking every 16th sample instead of averaging improves overall accuracy:

| Method | Overall | Wake | N1 | N2 | N3 | REM |
|--------|---------|------|----|----|----|----|
| Averaging | 83.2% | 66.3% | 72.7% | 96.9% | 99.3% | 70.6% |
| Every 16th | **86.8%** | **77.2%** | 54.5% | 94.7% | 98.6% | 64.7% |

**Tradeoff Analysis:**
- Wake detection: +11% improvement (main benefit)
- N1 detection: -18% (worse, but only 11 samples)
- N2/N3: Slight decrease but still excellent (>94%)
- REM: -6% (worse)

**Recommendation**: If Wake detection is important for your use case, switch to "every 16th sample" downsampling. This is also simpler to implement on Teensy (just skip samples instead of averaging).

---

## 500Hz Intermediate Sampling Frequency Tests

We tested whether using 500Hz as an intermediate sampling frequency (matching a colleague's original implementation) would improve agreement.

### Pipeline Variations Tested

| Method | Description | Agreement |
|--------|-------------|-----------|
| Method A | 4kHz -> 500Hz (avg 8) -> filter -> 100Hz (avg 5) | 78.5% |
| Method B | 4kHz -> 500Hz (every 8th) -> filter -> 100Hz (avg 5) | 80.7% |
| Method C | 4kHz -> 500Hz (every 8th) -> filter -> 100Hz (every 5th) | **83.6%** |
| Method D | 4kHz -> 500Hz (avg 8) -> filter -> 100Hz (every 5th) | 82.2% |
| Method E | 4kHz -> 500Hz (avg 8) -> filter -> 100Hz (resample) | 82.5% |

### 500Hz vs 250Hz Comparison

| Method | Agreement | Wake | N2 | N3 |
|--------|-----------|------|----|----|
| 250Hz averaging (current) | 83.2% | 66% | 97% | 99% |
| 500Hz Method C (best 500Hz) | 83.6% | 69% | 96% | 99% |
| **250Hz every 16th (best overall)** | **86.8%** | **77%** | 95% | 99% |

### Conclusion

The 500Hz intermediate sampling frequency does **NOT** improve over the best 250Hz method. The "every 16th sample" approach at 250Hz remains the best method found with 86.8% agreement.

---

## Model Input Specification

**Important**: The model uses **FLOAT32** inputs, not INT8 quantized.

| Input | Shape | Type | Quantization |
|-------|-------|------|--------------|
| EEG data | [1, 1, 3000, 1] | float32 | None (scale=0.0) |
| Epoch index | [1, 1] | float32 | None (scale=0.0) |
| Output | [1, 5] | float32 | None |

Normalized float values are passed directly to the model without any INT8 conversion.

---

## Files and Tools

| File | Purpose |
|------|---------|
| `tools/test_prediction_agreement.py` | Phase 1: Python vs reference prediction test |
| `tools/compare_teensy_python.py` | Phase 2: Teensy vs Python checkpoint comparison |
| `tools/compare_preprocessing.py` | Original preprocessing comparison tool |
| `src/test_eeg_playback.cpp` | Teensy playback with checkpoint debug output |
| `src/PreprocessingPipeline.cpp` | Current Teensy preprocessing (simple averaging) |

---

## Quick Reference Commands

```bash
# Run Python prediction agreement test
python tools/test_prediction_agreement.py

# Build and upload Teensy firmware
pio run --target upload

# Monitor Teensy serial output
pio device monitor --baud 115200

# Compare Teensy checkpoints with Python
python tools/compare_teensy_python.py teensy_debug.txt
```

---

## Decision Points

### Option A: Keep Simple Averaging (83.2% agreement)
- Current Teensy implementation
- Excellent deep sleep detection (N2: 97%, N3: 99%)
- Wake detection at 66%
- No code changes needed

### Option B: Switch to Every 16th Sample (86.8% agreement)
- Better overall accuracy (+3.6%)
- Much better Wake detection (77% vs 66%)
- Slightly worse sleep stage detection (still >94% for N2/N3)
- **Simpler to implement** - just skip samples instead of averaging
- Teensy change: In `PreprocessingPipeline.cpp`, change from averaging to decimation

### Recommendation
If Wake detection matters for your application (e.g., detecting when user falls asleep), switch to "every 16th sample". The implementation is actually simpler and gives better results.

### Code Change Required (if switching to every 16th sample)
In `src/PreprocessingPipeline.cpp`, change the downsampling logic:

```cpp
// CURRENT: Average every 16 samples
// for (int i = 0; i < 16; i++) {
//   sum_250hz += downsample_250hz_buffer_[i];
// }
// float sample_250hz = sum_250hz / 16.0f;

// NEW: Take every 16th sample (simpler and better accuracy)
float sample_250hz = downsample_250hz_buffer_[0];  // Just use first sample
```

### Alternative approaches (if needed):
- Ask colleague for exact preprocessing code used for reference data
- Retrain model using data preprocessed with our method

### Methods Ruled Out
- **500Hz intermediate sampling**: Tested 5 variations, best was 83.6% (worse than every 16th at 86.8%)
- **scipy.signal.decimate**: Both FIR and IIR variants performed worse
- **Zero-phase filtering (filtfilt)**: Much worse at 78%
- **Filtering at 4kHz first**: Unstable, 43% agreement
