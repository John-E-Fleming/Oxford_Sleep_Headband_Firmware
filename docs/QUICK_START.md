# Quick Start Guide

Get the sleep headband firmware running in under 10 minutes.

---

## Prerequisites Checklist

Before starting, ensure you have:

- [ ] **Teensy 4.1** microcontroller board
- [ ] **microSD card** (FAT32 formatted, 8GB+ recommended)
- [ ] **USB cable** (data-capable, not charge-only)
- [ ] **VS Code** installed ([download](https://code.visualstudio.com/))
- [ ] **PlatformIO extension** installed in VS Code
- [ ] **Teensyduino** installed (Windows only, [download](https://www.pjrc.com/teensy/td_download.html))

---

## Installation Steps

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd sleep_headband_firmware
```

> **Windows users:** Use a short path like `C:\Projects\sleep_firmware` to avoid path length issues.

### Step 2: Open in VS Code

1. Open VS Code
2. Click **File > Open Folder**
3. Select the `sleep_headband_firmware` folder
4. Wait for PlatformIO to initialize (may take 1-2 minutes first time)

### Step 3: Install Dependencies

Dependencies install automatically on first build. No manual steps needed.

---

## Two Paths: Choose Your Mode

### Path A: Validation Mode (Recommended First)

Test the system with recorded EEG data to verify everything works.

#### A1. Prepare SD Card

Format your microSD as FAT32 and add these files:

```
SD Card Root/
├── config.txt                    (configuration file)
├── SdioLogger_miklos_night_2.bin (EEG recording)
└── data/
    └── reference_predictions.csv (for validation)
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
```

#### A2. Build and Upload

```bash
# Build the firmware
pio run

# Upload to Teensy
pio run --target upload
```

Or use the PlatformIO sidebar in VS Code.

#### A3. Monitor Output

```bash
pio device monitor
```

**Expected output:**
```
=== EEG File Playback Test ===
SD card initialized
Configuration loaded successfully
ML inference ready (real model)
Starting EEG playback...

[TIMING] Data loading took 330 ms
[TIMING] Inference took 130 ms
[Epoch 0] Stage: N2 (87.3%)
```

#### A4. Verify It's Working

You should see:
- SD card initialization success
- ML inference ready message
- Sleep stage predictions every 30 seconds
- ~81% agreement with reference predictions (if validation enabled)

---

### Path B: Real-Time Mode

Use with live ADS1299 EEG hardware.

#### B1. Configure Hardware

Connect ADS1299 to Teensy 4.1:

| Signal | Teensy Pin | Notes |
|--------|-----------|-------|
| CS | 7 | Chip Select |
| DRDY | 22 | Data Ready (interrupt) |
| START | 15 | Start conversion |
| PWDN | 14 | Power down |
| MOSI | 11 | SPI data out |
| MISO | 12 | SPI data in |
| SCK | 13 | SPI clock |

#### B2. Switch Build Mode

Edit `platformio.ini` and comment/uncomment the appropriate `build_src_filter` line:

```ini
; Comment out PLAYBACK MODE line
; build_src_filter = +<*> -<main_realtime_inference.cpp> ...

; Uncomment REAL-TIME MODE line
build_src_filter = +<*> -<main_playback_inference.cpp> ...
```

#### B3. Build and Upload

```bash
pio run --target clean  # Clean old build
pio run                  # Build real-time mode
pio run --target upload  # Upload to Teensy
```

#### B4. Expected Output

```
===========================================
Sleep Headband - Real-Time Inference Mode
===========================================

ADS1299 initialized successfully
Data acquisition started at 4000Hz
ML inference ready (real model)

[Epoch 0] Stage: WAKE (91.2%) | Inference: 130ms | Time: 30.0s
```

---

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| Build fails with path errors | Move project to shorter path (e.g., `C:\Projects\`) |
| "SD card initialization failed" | Check card is FAT32, properly inserted |
| No serial output | Check COM port, baud rate (115200), reset Teensy |
| "Failed to initialize ADS1299" | Check wiring, power supply |
| Low validation agreement (<70%) | Check TFLite library version, filter coefficients |

---

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand the system design
- Read [VALIDATION_GUIDE.md](VALIDATION_GUIDE.md) for detailed validation instructions
- Read [HARDWARE_SETUP.md](HARDWARE_SETUP.md) for ADS1299 wiring details
- Read [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues

---

## Interactive Commands (Playback Mode)

While running in playback mode, press these keys in the serial monitor:

| Key | Action |
|-----|--------|
| `p` | Toggle serial plotting |
| `i` | Toggle ML inference |
| `s` | Show statistics |
| `d` | Toggle debug logging |
| `r` | Restart playback |
