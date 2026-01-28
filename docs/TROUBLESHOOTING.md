# Troubleshooting Guide

Solutions to common problems with the sleep headband firmware.

---

## Build Errors

### "Path too long" / File not found errors (Windows)

**Symptom:** Build fails with path-related errors

**Solution:**
1. Move project to shorter path: `C:\Projects\sleep_firmware`
2. Or enable long paths in Windows:
   ```powershell
   New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
                    -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
   ```
3. Restart computer

### "Library not found" errors

**Symptom:** Missing library during compilation

**Solution:**
1. Delete `.pio` folder
2. Rebuild: `pio run`
3. If persists, check internet connection

### Compilation errors in TensorFlow Lite

**Symptom:** Errors in TFLite header files

**Solution:**
1. Verify library version in `platformio.ini`:
   ```ini
   https://github.com/tensorflow/tflite-micro-arduino-examples.git#2be8092d9f167b1473f072ff5794364819df8b52
   ```
2. Delete `.pio/libdeps/teensy41/Arduino_TensorFlowLite`
3. Rebuild

### "Multiple definition" linker errors

**Symptom:** Linker complains about duplicate symbols

**Solution:**
1. Check `build_src_filter` in platformio.ini
2. Ensure only ONE main file is included
3. Clean and rebuild: `pio run --target clean && pio run`

---

## SD Card Problems

### "SD card initialization failed"

**Possible causes and solutions:**

| Cause | Solution |
|-------|----------|
| Wrong format | Reformat as FAT32 (not exFAT) |
| Card not inserted | Push card in firmly until it clicks |
| Damaged card | Try a different SD card |
| Card too fast/slow | Use Class 10, UHS-I card |

### "File not found" / "Could not open data file"

**Check:**
1. Filename matches exactly (case-sensitive on some systems)
2. File is in SD card root, not a subfolder
3. config.txt `datafile=` parameter is correct
4. File isn't corrupted (check size)

### Slow SD card performance

**Symptoms:** Long delays, choppy playback

**Solutions:**
1. Use faster SD card (Class 10 minimum)
2. Defragment files (copy to PC, format, copy back)
3. Use SDIO mode (default) not SPI

---

## Serial Monitor Issues

### No output in serial monitor

**Checklist:**
- [ ] Correct COM port selected
- [ ] Baud rate set to 115200
- [ ] USB cable is data-capable (not charge-only)
- [ ] Press reset button on Teensy after upload
- [ ] Wait 5 seconds for initialization

### Garbled output

**Cause:** Baud rate mismatch

**Solution:** Set serial monitor to 115200 baud

### Output stops unexpectedly

**Possible causes:**
1. End of file reached (normal for playback)
2. Firmware crash (check for error messages)
3. USB disconnection

---

## Hardware Problems

### "Failed to initialize ADS1299"

**Wiring checklist:**
- [ ] CS pin connected (default: pin 7)
- [ ] DRDY pin connected (default: pin 22)
- [ ] SPI pins: MOSI(11), MISO(12), SCK(13)
- [ ] Power: 3.3V to DVDD, 5V to AVDD
- [ ] Ground connected

**Software checklist:**
- [ ] Correct pins defined in `ADS1299_Custom.h`
- [ ] Building in real-time mode (not playback)

### No data from ADS1299

**Check:**
1. START pin is connected and configured
2. PWDN pin is HIGH (not in power-down mode)
3. Electrodes connected or inputs shorted for testing

### Very noisy EEG signal

**Solutions:**
1. Improve electrode contact
2. Add shielding to cables
3. Move away from AC power sources
4. Check grounding

---

## ML Inference Problems

### "Failed to initialize ML inference"

**Possible causes:**
1. **Insufficient memory** - Reduce tensor arena size
2. **Corrupted model** - Regenerate model.h from .tflite
3. **Wrong library** - Check TFLite library version

### Wrong or random predictions

**Check:**
1. Bipolar channels correct (positive - negative)
2. Sample rate matches (4000 Hz expected)
3. Gain/vref settings correct in config.txt
4. Preprocessing pipeline enabled

### Inference takes too long (> 500ms)

**Solutions:**
1. Ensure tensor arena in internal RAM (not PSRAM)
2. Check for excessive serial printing
3. Verify not using DEBUG mode

---

## Validation Problems

### Low agreement (< 70%)

**Systematic debugging:**

1. **Check library version:**
   ```bash
   cat .pio/libdeps/teensy41/Arduino_TensorFlowLite/library.json
   ```
   Should match specified commit hash.

2. **Check filter coefficients:**
   Compare `TrainingBandpassFilter.h` with Python training code.

3. **Check preprocessing pipeline:**
   Enable checkpoint debugging, compare with Python.

4. **Check model:**
   Verify `include/model.h` matches Python model.

### "Validation mode enabled but failed to load"

1. Create `data/` folder on SD card
2. Place `reference_predictions.csv` in `data/` folder
3. Check CSV format matches expected schema

### MSE very high

Usually indicates normalization mismatch:
1. Verify Z-score is computed per-window
2. Check for NaN/Inf values in data
3. Verify epoch index is being passed to model

---

## Performance Problems

### Playback slower than real-time

**Check:**
1. Serial printing disabled (`ENABLE_SAMPLE_PRINTING = false`)
2. Fast playback mode enabled (`FAST_PLAYBACK = true`)
3. SD card is Class 10 or faster

### Memory overflow / crashes

**Symptoms:** Random resets, garbled output

**Solutions:**
1. Reduce buffer sizes
2. Use EXTMEM for large arrays
3. Check stack usage with `printf("Free RAM: %d\n", freeMemory());`

---

## Quick Diagnostic Commands

### Check build configuration

```bash
pio run --target envdump
```

### Clean build

```bash
pio run --target clean
pio run
```

### Check serial ports

```bash
pio device list
```

### Monitor with timestamp

```bash
pio device monitor --filter time
```

---

## Getting Help

If problems persist:

1. **Collect information:**
   - Full error message
   - platformio.ini contents
   - Serial monitor output
   - Hardware configuration

2. **Check existing documentation:**
   - [ARCHITECTURE.md](ARCHITECTURE.md)
   - [VALIDATION_GUIDE.md](VALIDATION_GUIDE.md)
   - `CLAUDE.md` (AI assistant context)

3. **External resources:**
   - [PlatformIO docs](https://docs.platformio.org)
   - [Teensy forum](https://forum.pjrc.com)
   - [TFLite Micro examples](https://github.com/tensorflow/tflite-micro-arduino-examples)
