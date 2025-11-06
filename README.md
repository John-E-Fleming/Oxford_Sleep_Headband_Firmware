# Sleep Headband Firmware

**Real-time EEG sleep stage classification on Teensy 4.1**

This firmware implements a complete EEG signal-processing pipeline for automated **sleep stage classification** on the **Teensy 4.1** microcontroller. It reads multi-channel EEG data from SD card files, performs bipolar derivation and filtering, and runs a **CNN** model for real-time inference of **Wake, Light, Deep, and REM** stages.

---

## Table of Contents

- [Quick Start for Beginners](#quick-start-for-beginners)
  - [Prerequisites](#prerequisites)
  - [Installation Steps](#installation-steps)
  - [First Build and Upload](#first-build-and-upload)
- [Current Project Status](#current-project-status)
  - [Recent Updates](#recent-updates)
  - [Validation Mode](#validation-mode)
  - [Known Issues](#known-issues)
- [Overview](#overview)
- [Features](#features)
- [Hardware Requirements](#hardware-requirements)
- [Signal Processing Pipeline](#signal-processing-pipeline)
- [Configuration](#configuration)
  - [SD Card Setup](#sd-card-setup)
  - [Configuration Parameters](#configuration-parameters)
- [Binary File Format](#binary-file-format)
- [Build Commands](#build-commands)
- [Serial Monitoring](#serial-monitoring)
- [Development Modes](#development-modes)
- [Validation Mode Guide](#validation-mode-guide)
- [Preparing Data Files](#preparing-data-files)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Quick Start for Beginners

### Prerequisites

This project uses **PlatformIO** with **VS Code** for development. Follow these steps to get started:

#### 1. Install VS Code

- Download and install [Visual Studio Code](https://code.visualstudio.com/)
- VS Code is a free, lightweight code editor that works on Windows, macOS, and Linux

#### 2. Install PlatformIO Extension

1. Open VS Code
2. Click the **Extensions** icon in the left sidebar (or press `Ctrl+Shift+X`)
3. Search for "**PlatformIO IDE**"
4. Click **Install** on the PlatformIO IDE extension
5. Wait for installation to complete (may take a few minutes)
6. Restart VS Code when prompted

#### 3. Install Teensyduino (Windows)

The Teensy 4.1 requires Teensyduino for uploading firmware:

1. Download [Teensyduino](https://www.pjrc.com/teensy/td_download.html)
2. Run the installer and follow the prompts
3. This installs the necessary USB drivers for Teensy boards

### Installation Steps

#### 1. Clone or Download the Repository

```bash
# Clone the repository
git clone <repository-url>
cd sleep_firmware

# OR download as ZIP and extract to C:\Projects\sleep_firmware
```

> **Note for Windows users:** Use a short path like `C:\Projects\sleep_firmware` to avoid Windows path length issues (260 character limit).

#### 2. Open Project in VS Code

1. Open VS Code
2. Click **File > Open Folder**
3. Navigate to and select the `sleep_firmware` folder
4. PlatformIO will automatically detect the project and install dependencies

#### 3. Install Dependencies

PlatformIO will automatically install the required libraries when you first build:

- **RingBuf** (circular buffer library)
- **SD** (SD card support)
- **TensorFlow Lite Micro** (machine learning inference)
- **SdFat** (enhanced SD card library)

This happens automatically during the first build - no manual steps needed.

### First Build and Upload

#### 1. Prepare Your SD Card

Format a microSD card as **FAT32** and create a `config.txt` file in the root:

```ini
datafile=SdioLogger_miklos_night_2.bin
sample_rate=4000
channels=9
format=int32
gain=24
vref=4.5
bipolar_channel_positive=0
bipolar_channel_negative=6
ml_target_sample_rate=100
ml_window_seconds=30
```

Add your EEG data file (`.bin`) to the SD card root.

#### 2. Build the Firmware

In VS Code with PlatformIO:

1. Open the **PlatformIO** sidebar (alien icon on the left)
2. Expand **teensy41** environment
3. Click **Build** (or use terminal: `pio run`)

The first build takes 2-5 minutes as it compiles TensorFlow Lite.

#### 3. Upload to Teensy

1. Connect Teensy 4.1 to your computer via USB
2. Press the **Upload** button in PlatformIO (or run `pio run --target upload`)
3. The Teensy Loader window will open automatically
4. Press the **reset button** on Teensy if needed
5. Wait for upload to complete

#### 4. Monitor Serial Output

```bash
# Open serial monitor
pio device monitor

# OR click "Monitor" in PlatformIO sidebar
```

You should see output like:
```
=== EEG Playback Starting ===
Loading configuration from SD card...
Playback initialized successfully
Processing EEG data...
```

---

## Current Project Status

### Recent Updates

**Date:** November 6, 2025

#### TensorFlow Lite Library Fix - COMPLETED

The firmware has been updated to use the **correct TensorFlow Lite library version**:

- **Previous:** `spaziochirale/Chirale_TensorFLowLite@^2.0.0` (incorrect)
- **Current:** `tensorflow/tflite-micro-arduino-examples` commit `2be8092d9f167b1473f072ff5794364819df8b52` (correct)

This fix ensures the firmware predictions match the Python reference implementation used during model training.

**Changes Made:**
- Updated `platformio.ini` to reference the correct TFLite library
- Fixed MicroInterpreter API compatibility (added nullptr parameter)
- Moved project to shorter path (`C:\Projects\sleep_firmware`) to avoid Windows long path issues
- Successfully compiled and uploaded with correct library

### Validation Mode

**Status:** ENABLED by default

Validation mode compares firmware predictions against Python reference predictions to verify the implementation is correct.

**How it works:**
1. Firmware loads `reference_predictions.csv` from SD card
2. For each 4-second epoch, compares firmware prediction to reference
3. Reports agreement percentage every 10 epochs
4. Final summary shows overall agreement (expected ~94.1%)

**To use validation mode:**

1. Generate reference predictions using the Python validation notebook
2. Copy `reference_predictions.csv` to SD card root (alongside data file)
3. Upload firmware (validation mode is enabled in `platformio.ini` line 29)
4. Monitor serial output for validation statistics

See `VALIDATION_MODE_GUIDE.md` for detailed instructions.

### Known Issues

**None currently.** The TensorFlow Lite library mismatch has been resolved.

**Previous issues (resolved):**
- ✓ Windows long path errors during library installation
- ✓ TFLite library version mismatch causing prediction differences
- ✓ Case-sensitive stage comparison in validation ("WAKE" vs "Wake")

---

## Overview

- **Goal:** Embedded, real-time sleep stage classification from EEG on a resource-constrained MCU
- **Input:** Multi-channel EEG (interleaved `int32`) read from SD card at 4000 Hz
- **Processing:** Bipolar derivation → downsampling → band-pass (0.5–40 Hz) → windowing → normalization → CNN
- **Output:** Sleep stage label and confidence via serial; optional validation against reference predictions

The firmware supports multiple operational modes:
- **File playback mode** (default): Stream EEG from SD card for testing
- **Live acquisition mode**: Real-time data from ADS1299 chips
- **Validation mode**: Compare predictions against Python reference

---

## Features

- **Real-time EEG processing:** Configurable bipolar channel derivation
- **High-performance filtering:** 0.5–40 Hz **6th-order Butterworth** IIR filter optimized for training data
- **Machine Learning:** CNN-based classifier for **Wake / Light / Deep / REM**
- **Validation system:** Verify firmware matches Python reference implementation
- **Config on SD:** Edit `config.txt` without reflashing firmware
- **Memory-efficient:** Circular buffers and EXTMEM usage for Teensy 4.1
- **Serial monitoring:** Live stage predictions, confidence scores, and validation statistics
- **Multiple modes:** File playback, live acquisition, or validation testing

---

## Hardware Requirements

- **Teensy 4.1** (600 MHz ARM Cortex-M7, 8MB PSRAM)
- **microSD card** (formatted FAT32) with EEG data and `config.txt`
- **USB serial** connection for logs and monitoring

Optional:
- **ADS1299 EEG acquisition boards** for live data capture
- External power supply for long-duration recordings

---

## Signal Processing Pipeline

```
Raw EEG (4000 Hz, 9 channels from SD card)
   └─► Bipolar derivation: ch_pos − ch_neg (e.g., F3-M2)
         └─► Downsample to 100 Hz
               └─► 0.5–40 Hz band-pass (6th-order Butterworth IIR)
                     └─► 30-second windowing (3000 samples @ 100 Hz)
                           └─► Adaptive z-score normalization
                                 └─► CNN sleep-stage classification
                                       └─► Stage + confidence over serial
```

**Sleep stages:** Wake, Light Sleep (N1-N2), Deep Sleep (N3/SWS), REM

**Key Features:**
- Uses `TrainingBandpassFilter` with coefficients matching Python training pipeline
- Circular buffering for continuous data streams
- TensorFlow Lite Micro for on-device inference

---

## Configuration

### SD Card Setup

1. **Format** the microSD to **FAT32**
2. Create a `config.txt` file in the SD root
3. Add your EEG binary file (`.bin`) to the SD root
4. **(Optional)** Add `reference_predictions.csv` for validation mode

Example SD card structure:
```
SD Card Root:
├── config.txt
├── SdioLogger_miklos_night_2.bin
└── reference_predictions.csv          (optional, for validation)
```

Example `config.txt`:
```ini
datafile=SdioLogger_miklos_night_2.bin
sample_rate=4000
channels=9
format=int32
gain=24
vref=4.5
bipolar_channel_positive=0
bipolar_channel_negative=6
ml_target_sample_rate=100
ml_window_seconds=30
```

### Configuration Parameters

| Parameter                  | Description                                       | Example          |
| -------------------------- | ------------------------------------------------- | ---------------- |
| `datafile`                 | EEG binary filename on SD root                    | `data.bin`       |
| `sample_rate`              | Input sampling rate (Hz)                          | `4000`           |
| `channels`                 | Number of channels in the file                    | `9`              |
| `format`                   | Data format (`int32` / `float32` / `int16`)       | `int32`          |
| `gain`                     | Analog gain used in ADC front-end                 | `24`             |
| `vref`                     | ADC reference voltage (V)                         | `4.5`            |
| `bipolar_channel_positive` | Positive channel index (0-based)                  | `0` (F3)         |
| `bipolar_channel_negative` | Negative channel index (0-based)                  | `6` (M2)         |
| `ml_target_sample_rate`    | Processing rate for ML block (Hz)                 | `100`            |
| `ml_window_seconds`        | CNN inference window length (s)                   | `30`             |

> **Note:** The firmware expects `int32` device counts. Conversion to µV: `uV = counts * ((2*vref)/gain) / 2^24 * 1e6`

---

## Binary File Format

- **Format:** Interleaved samples across channels
- **Structure:** `t0: ch0, ch1, ..., ch8, t1: ch0, ch1, ..., ch8, ...`
- **Endianness:** Little-endian (native ARM)
- **Data type:** `int32` (32-bit signed integers)
- **Channels:** 9 channels (configurable)
- **Sample rate:** 4000 Hz (configurable)

The firmware performs bipolar derivation: `output = channel[positive] - channel[negative]`

---

## Build Commands

```bash
# Build the project
pio run

# Clean build artifacts
pio run --target clean

# Upload to Teensy 4.1
pio run --target upload

# Open serial monitor
pio device monitor

# Build and upload in one command
pio run --target upload && pio device monitor
```

### Build Configuration

The project uses `build_src_filter` in `platformio.ini` to select which main file to compile:

**Current (default): File playback mode with validation**
```ini
build_src_filter = +<*> -<main.cpp> -<main_ml.cpp> -<debug_ml_pipeline.cpp> -<sd_test.cpp> -<simple_test.cpp> -<example_print_to_serial.cpp> -<model_placeholder.cpp> -<batch_inference_test.cpp>
```
Uses `test_eeg_playback.cpp` as main - streams EEG from SD card, supports ML inference and validation.

**Other available modes:**

- **Live ADS1299 acquisition:** Change filter to use `main_ml.cpp`
- **Basic SD streaming:** Change filter to use `main.cpp`

---

## Serial Monitoring

Connect to the Teensy via USB serial at **115200 baud**.

### Normal Operation

```
=== EEG Playback Starting ===
Config loaded: datafile=SdioLogger_miklos_night_2.bin, fs=4000Hz
ML inference initialized (100 Hz, 30s windows)
Processing epoch 1... Stage: LIGHT, Confidence: 0.87
Processing epoch 2... Stage: LIGHT, Confidence: 0.92
```

### Validation Mode

```
=== VALIDATION MODE ENABLED ===
Loaded 960 reference predictions from SD card
Validation ready with 960 reference predictions
===============================

Epoch 1: FW=LIGHT (0.87), REF=Light (0.85) - MATCH ✓
Epoch 2: FW=LIGHT (0.92), REF=Light (0.91) - MATCH ✓
...
Progress: Agreement 9/10 (90.0%)
...
========================================
VALIDATION SUMMARY
========================================
Total epochs compared: 960
Exact stage matches: 903/960 (94.1%)
Mean probability MSE: 0.002145
========================================
```

Expected agreement: **~94.1%** (matches Python validation notebook)

---

## Development Modes

### File Playback Mode (Default)

**Use case:** Test ML pipeline on pre-recorded data

**Main file:** `src/test_eeg_playback.cpp`

**Features:**
- Streams EEG from SD card at correct timing (4000 Hz)
- Supports ML inference every 4 seconds
- Interactive commands: `p` (toggle plotting), `i` (toggle inference), `s` (statistics)
- Validation against reference predictions

### Live Acquisition Mode

**Use case:** Real-time EEG capture and classification

**Main file:** `src/main_ml.cpp`

**Features:**
- Interfaces with ADS1299 chips via SPI
- Real-time data acquisition at 4000 Hz
- Continuous ML inference
- Optional data logging to SD card

To switch to this mode:
1. Update `build_src_filter` in `platformio.ini` to exclude `test_eeg_playback.cpp` and include `main_ml.cpp`
2. Configure ADS1299 pins in `include/ADS1299_Custom.h`
3. Rebuild and upload

### Validation Mode

**Use case:** Verify firmware matches Python reference

**How to enable:** Already enabled by default (`-DENABLE_VALIDATION_MODE` in `platformio.ini` line 29)

**Requirements:**
- `reference_predictions.csv` on SD card
- Generated using Python validation notebook
- Contains 960 epochs of predictions (stage labels + probabilities)

See `VALIDATION_MODE_GUIDE.md` for complete instructions.

---

## Validation Mode Guide

### Purpose

Validation mode verifies that the firmware preprocessing and ML inference match the Python reference implementation. This ensures the embedded system produces the same predictions as the training environment.

### Setup

1. **Generate reference predictions** using the Python notebook:
   ```python
   # In validation notebook
   python_predictions.to_csv('reference_predictions.csv', index=False)
   ```

2. **Copy to SD card:**
   ```
   SD Card:
   ├── config.txt
   ├── SdioLogger_miklos_night_2.bin
   └── reference_predictions.csv    ← Add this file
   ```

3. **Ensure validation is enabled** in `platformio.ini`:
   ```ini
   build_flags =
       ...
       -DENABLE_VALIDATION_MODE
   ```

4. **Upload and monitor:**
   ```bash
   pio run --target upload
   pio device monitor
   ```

### Expected Results

- **Agreement:** ~94.1% exact stage matches
- **MSE:** ~0.002 for probability distributions
- **Mismatches:** Usually at stage boundaries (inherent ambiguity)

### Troubleshooting Validation

**Low agreement (<90%)?**
- Verify TensorFlow Lite library version (should be commit 2be8092d)
- Check filter coefficients match training (use `TrainingBandpassFilter`)
- Ensure model file (`include/model.h`) matches Python model
- Verify SD card files are correct versions

**File not found?**
- Check `reference_predictions.csv` is in SD card root
- Verify FAT32 format
- Check filename spelling (case-sensitive on some systems)

---

## Preparing Data Files

If your source recordings are at high rates (e.g., 4000 Hz), you can use them directly. The firmware handles downsampling internally.

### Using Pre-recorded Data

1. Export from your acquisition system as **int32 binary** format
2. Ensure data is **interleaved by channel**: `ch0, ch1, ..., ch8, ch0, ch1, ...`
3. Set correct parameters in `config.txt`:
   - `sample_rate`: Original recording rate (e.g., 4000)
   - `channels`: Number of channels (e.g., 9)
   - `format`: Data type (e.g., int32)

### Example: Downsampling with Python

If you need to pre-process data:

```python
import numpy as np
from scipy.signal import decimate

# Load high-rate data (4000 Hz)
data = np.fromfile('raw_4000Hz.bin', dtype=np.int32)
data = data.reshape(-1, 9)  # 9 channels

# Downsample to 100 Hz (optional - firmware can do this)
downsampled = decimate(data, q=40, axis=0)

# Save as int32
downsampled.astype(np.int32).tofile('data_100Hz.bin')
```

Then update `config.txt`:
```ini
datafile=data_100Hz.bin
sample_rate=100
```

---

## Project Structure

```
sleep_firmware/
├── .pio/                          # PlatformIO build artifacts (auto-generated)
├── .vscode/                       # VS Code settings
├── data/                          # Sample data files
├── example_datasets/              # Debug and test datasets
│   └── debug/                     # Colleague's reference data for validation
├── include/                       # Header files
│   ├── ADS1299_Custom.h          # EEG acquisition hardware
│   ├── EEGProcessor.h            # Signal processing
│   ├── MLInference.h             # TensorFlow Lite integration
│   ├── PreprocessingPipeline.h   # Complete pipeline
│   ├── TrainingBandpassFilter.h  # Correct filter coefficients
│   ├── ValidationReader.h        # Validation system
│   └── model.h                   # TFLite model (converted from .tflite)
├── src/                          # Source files
│   ├── test_eeg_playback.cpp    # Main file (file playback mode)
│   ├── main_ml.cpp              # Live ADS1299 acquisition
│   ├── main.cpp                 # Basic SD streaming
│   ├── EEGProcessor.cpp         # Signal processing implementation
│   ├── MLInference.cpp          # ML inference implementation
│   ├── ValidationReader.cpp     # Validation logic
│   └── ...                      # Other implementation files
├── platformio.ini               # PlatformIO configuration
├── CLAUDE.md                    # Project guidance for AI assistants
├── PENDING_TFLITE_FIX.md       # TFLite library fix documentation (resolved)
├── VALIDATION_MODE_GUIDE.md    # Detailed validation instructions
├── FILTER_UPDATE_SUMMARY.md    # Filter coefficient update notes
└── README.md                   # This file
```

### Key Files

- **platformio.ini**: Build configuration, library dependencies, build flags
- **include/model.h**: TensorFlow Lite model (auto-generated from .tflite file)
- **include/TrainingBandpassFilter.h**: Filter coefficients matching Python training
- **src/test_eeg_playback.cpp**: Main file for SD card playback and validation
- **VALIDATION_MODE_GUIDE.md**: Complete validation mode documentation

---

## Troubleshooting

### Build Issues

**Long path errors (Windows)?**
- Move project to shorter path: `C:\Projects\sleep_firmware`
- Enable long paths in Windows (requires admin):
  ```powershell
  New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
                   -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
  ```
- Restart computer after enabling

**Library installation fails?**
- Clear PlatformIO cache: Delete `.pio` folder and rebuild
- Check internet connection (PlatformIO downloads libraries from GitHub)
- Verify `platformio.ini` has correct library URLs

**Compilation errors?**
- Ensure TensorFlow Lite library is correct version (commit 2be8092d)
- Check all files are saved
- Try clean build: `pio run --target clean && pio run`

### Runtime Issues

**No data found on boot?**
- Confirm SD card is inserted before powering on
- Verify `config.txt` and data file are on SD card root
- Check SD card is formatted FAT32 (not exFAT or NTFS)
- Try re-formatting SD card

**Validation shows low agreement?**
- Verify TFLite library version matches Python (commit 2be8092d)
- Check `reference_predictions.csv` is from matching dataset
- Ensure filter coefficients are correct (`TrainingBandpassFilter`)
- Verify model file matches Python model

**Firmware crashes or hangs?**
- Check serial monitor for error messages
- Verify sufficient power supply (USB may be insufficient during SD card access)
- Ensure EXTMEM is available (Teensy 4.1 with PSRAM chip)

**Wrong predictions or garbage output?**
- Verify bipolar channel indices match electrode positions
- Check `gain` and `vref` in config match acquisition settings
- Ensure data file format matches config (`int32` vs `float32`)
- Verify downsampling and filtering are working (check intermediate outputs)

**Serial monitor shows nothing?**
- Check correct COM port is selected
- Verify baud rate is 115200
- Press reset button on Teensy
- Check USB cable (some cables are charge-only)

### Getting Help

If you encounter issues not covered here:

1. Check `CLAUDE.md` for project-specific guidance
2. Review `VALIDATION_MODE_GUIDE.md` for validation issues
3. Check PlatformIO documentation: https://docs.platformio.org
4. Check Teensy forum: https://forum.pjrc.com
5. Open an issue in the repository with:
   - Full error message
   - Build output
   - `platformio.ini` contents
   - Serial monitor output

---

## License

TODO: Insert license here (e.g., MIT).

---

*Questions or contributions welcome—please open an issue or PR.*
