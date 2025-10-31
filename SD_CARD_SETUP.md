# SD Card Setup for ML Pipeline Testing

## Required Files

The following files need to be copied to your SD card:

**Source:** `data/example_datasets/debug/`
**Destination:** SD card root `/debug/` folder

### Files List:

1. **1_bandpassed_eeg_single_channel.npy** (23 MB)
   - Raw EEG data @ 100Hz, filtered 0.5-30Hz

2. **2_standardized_epochs.npy** (23 MB)
   - Z-score normalized 30-second epochs

3. **3_quantized_model_predictions.npy** (8 KB)
   - Model predictions (class indices)

4. **4_quantized_model_probabilities.npy** (19 KB)
   - Model output probabilities

**Optional (not required for testing):**
- `8_tflite_quantized_model.tflite` (55 KB) - Reference model
- `info.txt` - Documentation

## Setup Instructions

### Option 1: Using File Explorer (Windows)

1. **Insert SD card** into your computer's SD card reader
2. **Create debug folder** on SD card root:
   - Open SD card drive (e.g., `D:\` or `E:\`)
   - Create new folder called `debug`
3. **Copy files:**
   - Navigate to project folder: `sleep_headband_firmware\data\example_datasets\debug\`
   - Select files 1-4 (the `.npy` files)
   - Copy and paste into SD card `/debug/` folder
4. **Eject SD card** safely
5. **Insert into Teensy 4.1** built-in SD card slot

### Option 2: Using Command Line

```bash
# Windows
cd sleep_headband_firmware
mkdir E:\debug
copy "data\example_datasets\debug\*.npy" "E:\debug\"

# Replace E:\ with your SD card drive letter
```

```bash
# Linux/Mac
cd sleep_headband_firmware
mkdir /Volumes/SD_CARD/debug
cp data/example_datasets/debug/*.npy /Volumes/SD_CARD/debug/

# Replace /Volumes/SD_CARD with your SD card mount point
```

## Verify SD Card Contents

After copying, your SD card should have:
```
SD_CARD_ROOT/
└── debug/
    ├── 1_bandpassed_eeg_single_channel.npy (23,040,128 bytes)
    ├── 2_standardized_epochs.npy (23,040,128 bytes)
    ├── 3_quantized_model_predictions.npy (7,808 bytes)
    └── 4_quantized_model_probabilities.npy (19,328 bytes)
```

**Total size:** ~46 MB

## Testing

Once files are on the SD card:

1. **Insert SD card** into Teensy 4.1
2. **Upload firmware:**
   ```bash
   pio run --target upload
   ```
3. **Open Serial Monitor:**
   ```bash
   pio device monitor
   ```
4. **Check startup output** - Should show:
   ```
   Initializing SD card... OK

   Checking /debug folder contents:
     Found: 1_bandpassed_eeg_single_channel.npy (23040128 bytes)
     Found: 2_standardized_epochs.npy (23040128 bytes)
     Found: 3_quantized_model_predictions.npy (7808 bytes)
     Found: 4_quantized_model_probabilities.npy (19328 bytes)

   Checking for required files:
     ✓ /debug/1_bandpassed_eeg_single_channel.npy (23040128 bytes)
     ✓ /debug/2_standardized_epochs.npy (23040128 bytes)
     ✓ /debug/3_quantized_model_predictions.npy (7808 bytes)
     ✓ /debug/4_quantized_model_probabilities.npy (19328 bytes)

   ✓ All required files found on SD card
   ```

5. **Run tests:**
   - Send `t` for full test
   - Send `n` for normalization test
   - Send `i` for inference test

## Troubleshooting

### "ERROR: /debug folder not found on SD card"
- Make sure you created the `debug` folder (not `Debug` or `DEBUG`)
- SD card filesystem must be FAT32 or exFAT

### "ERROR: Some required files are missing"
- Check file names match exactly (case-sensitive on some systems)
- Ensure files copied completely (check file sizes)
- Files must be in `/debug/` folder, not SD card root

### "SD card initialization failed"
- Check SD card is properly inserted in Teensy 4.1
- Ensure SD card is formatted as FAT32 or exFAT
- Try different SD card if available
- Check Teensy 4.1 has built-in SD card slot (not all Teensys do)

### Files are too large / copy is slow
- This is normal - files are 23MB each
- USB 2.0 reader: ~2-5 minutes
- USB 3.0 reader: ~30-60 seconds
- Be patient and don't interrupt the copy

### "File already exists" or "Cannot overwrite"
- Delete old files from `/debug/` folder first
- Or rename old files before copying new ones

## SD Card Requirements

- **Capacity:** Minimum 64MB (recommended 1GB+)
- **Format:** FAT32 or exFAT
- **Speed Class:** Any (Class 4 or higher recommended)
- **Type:** SD, SDHC, or SDXC

## Notes

- The `.npy` files are NumPy array files (Python format)
- Teensy reads them as binary data (doesn't need NumPy library)
- Files contain float32 data with 128-byte NumPy header
- Do NOT modify or rename these files
- Keep originals safe in case you need to re-copy
