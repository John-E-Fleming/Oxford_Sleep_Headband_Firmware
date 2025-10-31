# Debug Logging Usage Guide

## Overview

The debug logging system captures intermediate pipeline outputs to CSV files for validation against your colleague's reference implementation. This allows you to verify that each stage of the preprocessing pipeline is working correctly.

---

## What Gets Logged

When debug logging is enabled, the firmware creates **4 CSV files** on the SD card:

### 1. `debug_preprocessed_100hz.csv`
- **Content**: 100Hz filtered data (after 4kHz→500Hz→100Hz downsampling and filtering)
- **Format**: Wide format - each row is an epoch (30 seconds = 3000 samples)
- **Columns**: `Epoch,Sample_0,Sample_1,...,Sample_2999`
- **Units**: Microvolts (μV)
- **Use**: Compare against colleague's filtered output

### 2. `debug_normalized.csv`
- **Content**: Z-score normalized data fed to ML model
- **Format**: Wide format - each row is an epoch
- **Columns**: `Epoch,Sample_0,Sample_1,...,Sample_2999`
- **Units**: Standardized (mean=0, std=1)
- **Use**: Verify normalization is working correctly

### 3. `debug_quantized.csv`
- **Content**: INT8 quantized model input (exactly what the model receives)
- **Format**: Wide format - each row is an epoch
- **Columns**: `Epoch,Sample_0,Sample_1,...,Sample_2999,Sample_3000` (includes epoch index)
- **Units**: INT8 values (-128 to 127)
- **Use**: Verify quantization matches colleague's implementation

### 4. `debug_model_output.csv`
- **Content**: CNN output probabilities for each inference
- **Format**: One row per inference
- **Columns**: `Epoch,Time_s,N3,N2,N1,REM,WAKE,Predicted_Stage`
- **Units**: Probabilities (0-1) for each sleep stage
- **Use**: Compare predictions against colleague's model outputs

---

## How to Use

### Step 1: Prepare the Teensy
1. Build and upload firmware: `pio run --target upload`
2. Insert SD card with `SdioLogger_miklos_night_2.bin` and `config.txt` (sample_rate=4000)
3. Connect to serial monitor: `pio device monitor`

### Step 2: Enable Debug Logging
1. Wait for firmware to initialize (you'll see "Debug logger ready")
2. Send command **`d`** to enable debug logging
3. You should see: `Debug logging: ENABLED`

### Step 3: Run Inference
1. Let the firmware run for 2-3 inference cycles (~60-90 seconds)
2. You'll see model predictions in the serial output
3. Debug files are being written to SD card in real-time

### Step 4: Collect Debug Files
1. Stop the firmware (reset or power off)
2. Remove SD card and copy these files to your computer:
   - `debug_preprocessed_100hz.csv`
   - `debug_normalized.csv`
   - `debug_model_output.csv`

### Step 5: Run Validation Script
```bash
python validate_pipeline.py --debug-dir /path/to/sd/card --reference-dir example_datasets/debug
```

The script will:
- Load your debug CSV files
- Load colleague's reference data (you need to implement loading logic)
- Calculate comparison statistics (MSE, correlation, MAE)
- Generate comparison plots
- Print validation report

---

## Serial Commands

| Command | Action |
|---------|--------|
| `p` | Toggle serial plotting ON/OFF |
| `i` | Toggle ML inference ON/OFF |
| `d` | **Toggle debug logging ON/OFF** |
| `s` | Show statistics (samples, inferences, file info) |
| `r` | Restart playback from beginning |

---

## Example Output

### CSV File Structure

**debug_preprocessed_100hz.csv:**
```csv
Epoch,Sample_0,Sample_1,Sample_2,...,Sample_2999
0,12.34,15.67,18.90,...,8.90
1,11.23,14.56,19.78,...,9.12
2,13.45,16.78,20.01,...,10.34
```

**debug_model_output.csv:**
```csv
Epoch,Time_s,N3,N2,N1,REM,WAKE,Predicted_Stage
0,30.0,0.05,0.10,0.15,0.20,0.50,WAKE
1,40.0,0.02,0.08,0.70,0.15,0.05,N1
2,50.0,0.01,0.75,0.15,0.05,0.04,N2
```

### Validation Report Example
```
==============================================================
PIPELINE VALIDATION REPORT
==============================================================

Data loaded:
  Your epochs: [0 1 2]
  Preprocessed shape: (3, 3000)
  Normalized shape: (3, 3000)

Comparison Statistics:
  mse: 0.000523
  rmse: 0.022869
  correlation: 0.998456
  mae: 0.015234
  max_error: 0.089123
  relative_error_percent: 0.234567

------------------------------------------------------------
ASSESSMENT:
✅ EXCELLENT: Correlation > 0.99 - pipelines match very well
✅ EXCELLENT: Relative error < 1%
------------------------------------------------------------
```

---

## Loading Your Colleague's Reference Data

You need to modify `validate_pipeline.py` to load your colleague's data format. Here's a template:

```python
def load_reference_data(reference_dir):
    """Load colleague's reference data"""
    ref_path = Path(reference_dir)
    data = {}

    # Example: Load colleague's filtered data (adjust to their format)
    filtered_file = ref_path / "filtered_output.csv"  # or .bin
    if filtered_file.exists():
        # If data is in ADC steps, convert to μV
        df = pd.read_csv(filtered_file)
        adc_data = df.values
        data['filtered_uv'] = adc_to_uv(adc_data)

    # Load their normalized data if available
    normalized_file = ref_path / "normalized_output.csv"
    if normalized_file.exists():
        df = pd.read_csv(normalized_file)
        data['normalized'] = df.values

    # Load their model predictions
    predictions_file = ref_path / "predictions.csv"
    if predictions_file.exists():
        data['predictions'] = pd.read_csv(predictions_file)

    return data
```

---

## Troubleshooting

### Issue: "Debug logger failed to initialize"
**Cause**: SD card not properly initialized
**Solution**:
- Check SD card is inserted
- Verify SD card is formatted (FAT32)
- Check SD card has free space

### Issue: Debug files are empty
**Cause**: Debug logging not enabled or no inferences ran
**Solution**:
- Send `d` command to enable
- Wait at least 30 seconds for first inference
- Check that ML inference is enabled (`i` command)

### Issue: Only model_output.csv is created (no preprocessed/normalized/quantized)
**Cause**: Not enough data in circular buffer yet
**Solution**:
- Wait for at least 30 seconds of data before first inference
- The `debug_buffer_full` flag must be true for preprocessed data
- Normalized and quantized data should always be logged once inference starts

### Issue: CSV files are corrupted
**Cause**: SD card removed while writing
**Solution**:
- Always stop firmware before removing SD card
- Wait a few seconds after last inference before power off

---

## File Size Estimates

For 10 minutes of recording (20 inferences at 30s intervals):

| File | Rows | Columns | Size |
|------|------|---------|------|
| `debug_preprocessed_100hz.csv` | 20 | 3001 | ~1.2 MB |
| `debug_normalized.csv` | 20 | 3001 | ~1.2 MB |
| `debug_quantized.csv` | 20 | 3002 | ~900 KB |
| `debug_model_output.csv` | 20 | 8 | ~2 KB |
| **Total** | | | **~3.3 MB** |

**Note**: Quantized CSV has 3002 columns (Epoch + 3000 EEG samples + 1 epoch index)

---

## Next Steps

1. ✅ Build and test firmware with debug logging
2. ⬜ Implement `load_reference_data()` in validation script for your colleague's format
3. ⬜ Run validation on debug dataset
4. ⬜ Compare results and identify any discrepancies
5. ⬜ If issues found, use staged validation (check each pipeline stage separately)

---

## Advanced: Per-Stage Validation

If full pipeline doesn't match, validate stages individually:

1. **Stage 1**: 4kHz→500Hz downsampling
   - Compare `debug_preprocessed_100hz.csv` samples 0-500 with colleague's first 500 samples

2. **Stage 2**: Beta bandpass + notch filtering
   - Should see 50Hz, 100Hz, 150Hz components removed

3. **Stage 3**: 500Hz→100Hz downsampling
   - Every 5th sample from colleague's 500Hz should match your 100Hz

4. **Stage 4**: Normalization
   - Check `debug_normalized.csv` has mean≈0, std≈1 for each epoch

5. **Stage 5**: Model predictions
   - Compare `debug_model_output.csv` against colleague's predictions
