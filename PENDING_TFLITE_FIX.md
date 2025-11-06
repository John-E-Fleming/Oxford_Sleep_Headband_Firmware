# TensorFlow Lite Library Version Fix - In Progress

**Status**: Windows long paths enabled, requires restart to complete fix

**Date**: 2025-11-05

---

## Issue Summary

The firmware is currently using the **wrong TensorFlow Lite library version**, causing prediction mismatches with the Python reference implementation.

**Root Cause**:
- Colleague generated `.tflite` model using: `tensorflow/tflite-micro-arduino-examples` commit `2be8092d9f167b1473f072ff5794364819df8b52`
- Firmware currently uses: `spaziochirale/Chirale_TensorFLowLite@^2.0.0` (different version)
- **Same model file + different TFLite runtime = different predictions**

**Why This Matters**:
- Python validation notebook shows 94.1% agreement (using correct library)
- Firmware predictions won't match reference predictions until we use the correct library
- Validation mode will show low agreement percentage with wrong library

---

## What Has Been Done

### 1. Windows Long Path Support Enabled ✓

The Windows registry has been updated to support long file paths:

```powershell
# Already executed successfully
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
                 -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Output confirmed:
```
LongPathsEnabled : 1
PSPath           : Microsoft.PowerShell.Core\Registry::HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem
```

### 2. Git Long Paths Enabled ✓

```bash
git config --global core.longpaths true
```

### 3. Case-Insensitive Stage Comparison Fixed ✓

Updated `src/ValidationReader.cpp` line 146 to use `strcasecmp()` instead of `strcmp()` to handle "WAKE" vs "Wake" case differences.

---

## What Needs to Be Done After Restart

### Step 1: Restart Computer (REQUIRED)

The Windows long path setting requires a restart to take full effect for PlatformIO's Python process.

### Step 2: Update platformio.ini

After restart, update `platformio.ini` to use the correct TensorFlow Lite library:

**Current (wrong) library**:
```ini
lib_deps =
	wizard97/RingBuf@^2.0.1
	arduino-libraries/SD@^1.3.0
	spaziochirale/Chirale_TensorFLowLite@^2.0.0
```

**Change to (correct) library**:
```ini
lib_deps =
	wizard97/RingBuf@^2.0.1
	arduino-libraries/SD@^1.3.0
	https://github.com/tensorflow/tflite-micro-arduino-examples.git#2be8092d9f167b1473f072ff5794364819df8b52
```

### Step 3: Clean and Rebuild

```bash
pio run --target clean
pio run
```

This should now succeed with long paths enabled after restart.

### Step 4: Upload and Test

```bash
pio run --target upload
```

Then open serial monitor to see validation results. Expected outcome: **~94.1% agreement** matching Python validation.

---

## Why Installation Failed Before Restart

**Error Encountered**:
```
shutil.Error: [('path/to/file', 'path/to/destination', '[Errno 22] Invalid argument')]
```

**Reason**: PlatformIO's Python process (`shutil.copytree`) doesn't respect the long path registry setting until the system is restarted. Even though the setting was enabled, running processes still use the old 260-character limit.

**File Paths Causing Issues**:
```
C:\Users\ndcm1133\OneDrive - Nexus365\Desktop\Oxford_Sleep_Headband\sleep_headband_firmware\
.pio\libdeps\teensy41\Arduino_TensorFlowLite\...
```

Base path is already 105 characters, and TFLite library has deep directory structures that exceed Windows' 260-character default limit.

---

## Alternative Fix (If Restart Doesn't Work)

If the build still fails after restart, you can move the project to a shorter path:

**From**:
```
C:\Users\ndcm1133\OneDrive - Nexus365\Desktop\Oxford_Sleep_Headband\sleep_headband_firmware
```

**To**:
```
C:\Projects\sleep_firmware
```

Then repeat Steps 2-4 above.

---

## Verification Steps

After successfully installing the correct library and uploading:

1. **Check Library Version**:
   Build output should show:
   ```
   Library Manager: Installing git+https://github.com/tensorflow/tflite-micro-arduino-examples.git#2be8092d9f167b1473f072ff5794364819df8b52
   ```

2. **Validation Results**:
   Serial monitor should show:
   ```
   === VALIDATION MODE ENABLED ===
   Validation ready with 960 reference predictions
   ===============================
   ```

3. **Expected Agreement**:
   Final summary should show ~94.1% agreement:
   ```
   ========================================
   VALIDATION SUMMARY
   ========================================
   Total epochs compared: 960
   Exact stage matches: 903/960 (94.1%)
   Mean probability MSE: 0.002145
   ========================================
   ```

---

## Current Configuration

**Validation Mode**: ✓ ENABLED
- `platformio.ini` line 29: `-DENABLE_VALIDATION_MODE`

**SD Card Files** (correct structure):
```
SD Card Root:
├── config.txt
├── data/
│   ├── SdioLogger_miklos_night_2.bin
│   └── reference_predictions.csv (960 epochs)
```

**ValidationReader Changes**: ✓ Applied
- Case-insensitive comparison implemented
- Memory allocation optimized (using `extmem_malloc` in `begin()`)

---

## Quick Start After Restart

```bash
# 1. Edit platformio.ini (change to correct TFLite library URL)
# 2. Clean build
pio run --target clean

# 3. Build with correct library (should work after restart)
pio run

# 4. Upload
pio run --target upload

# 5. Monitor validation results
# Open serial monitor in VS Code or:
pio device monitor
```

Look for:
- "Loaded 960 reference predictions"
- Periodic "Agreement: X/Y (Z%)" updates every 10 epochs
- Final validation summary with ~94.1% agreement

---

## Related Files

- `platformio.ini` - Line 18: Library dependency to change
- `platformio.ini` - Line 29: Validation mode flag (already enabled)
- `src/ValidationReader.cpp` - Validation comparison logic
- `include/ValidationReader.h` - Validation class definition
- `VALIDATION_MODE_GUIDE.md` - Full validation mode documentation
- `data/reference_predictions.csv` - Python reference predictions (on SD card)

---

## Contact/Notes

If agreement is still low after installing correct library:
1. Verify SD card files are correct
2. Check that model file (`include/model.h`) matches colleague's version
3. Review filter coefficients (should be using `TrainingBandpassFilter`)
4. Compare first few epoch predictions manually

The Python validation notebook already showed 94.1% agreement, so firmware should match once the correct TFLite library is installed.
