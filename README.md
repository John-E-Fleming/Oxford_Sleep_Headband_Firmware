# Sleep Headband Firmware

**Real-time EEG sleep stage classification on Teensy 4.1**

This firmware implements a complete EEG signal processing pipeline for automated sleep stage classification using TensorFlow Lite Micro. It supports both live EEG acquisition from ADS1299 hardware and playback from recorded files for validation.

---

## Quick Links

| Document | Description |
|----------|-------------|
| [Quick Start](docs/QUICK_START.md) | Get running in 10 minutes |
| [Architecture](docs/ARCHITECTURE.md) | System design and data flow |
| [Hardware Setup](docs/HARDWARE_SETUP.md) | ADS1299 wiring and configuration |
| [Validation Guide](docs/VALIDATION_GUIDE.md) | Testing and validation procedures |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common issues and solutions |
| [API Reference](docs/API_REFERENCE.md) | Code documentation |

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    SLEEP HEADBAND FIRMWARE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input (4kHz)          Processing              Output          │
│   ┌──────────┐    ┌─────────────────────┐    ┌──────────┐      │
│   │ ADS1299  │───>│ Bipolar derivation  │───>│  Sleep   │      │
│   │    or    │    │ 4kHz → 100Hz filter │    │  Stage   │      │
│   │ SD Card  │    │ 30s window + CNN    │    │ + Conf.  │      │
│   └──────────┘    └─────────────────────┘    └──────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Sleep Stages: Wake | N1 (Light) | N2 (Light) | N3 (Deep) | REM
```

---

## Validation Status

| Platform | Agreement | Notes |
|----------|-----------|-------|
| Python (reference) | 100% | Model inference correct |
| Python (our preprocessing) | 86.8% | Preprocessing verified |
| **Teensy (embedded)** | **81.4%** | 781/959 epochs match |

Performance: **~90x faster than real-time** (330ms per 30-second epoch)

---

## Operational Modes

The firmware supports three modes, selected via `platformio.ini`:

| Mode | Main File | Purpose |
|------|-----------|---------|
| **Playback** (default) | `main_playback_inference.cpp` | Validation with recorded data |
| **Real-Time** | `main_realtime_inference.cpp` | Live EEG from ADS1299 |
| **Streaming** | `main_data_streaming.cpp` | Hardware debugging (no ML) |

### Switching Modes

Edit `platformio.ini` and comment/uncomment the appropriate `build_src_filter` line:

```ini
; PLAYBACK MODE (DEFAULT)
build_src_filter = +<*> -<main_realtime_inference.cpp> -<main_data_streaming.cpp> ...

; REAL-TIME MODE
; build_src_filter = +<*> -<main_playback_inference.cpp> -<main_data_streaming.cpp> ...
```

---

## Quick Start

### Prerequisites

- Teensy 4.1 with PSRAM
- VS Code + PlatformIO extension
- microSD card (FAT32)
- Teensyduino (Windows)

### Build & Run

```bash
# Clone repository
git clone <repository-url>
cd sleep_headband_firmware

# Build
pio run

# Upload
pio run --target upload

# Monitor
pio device monitor
```

See [Quick Start Guide](docs/QUICK_START.md) for detailed instructions.

---

## Hardware Requirements

| Component | Description |
|-----------|-------------|
| Teensy 4.1 | ARM Cortex-M7 @ 600MHz, 8MB PSRAM |
| ADS1299 | 8-channel 24-bit EEG ADC (real-time mode) |
| microSD | FAT32, Class 10+ recommended |

### Default Pin Configuration

| Signal | Teensy Pin |
|--------|-----------|
| ADS1299 CS | 7 |
| ADS1299 DRDY | 22 |
| ADS1299 START | 15 |
| ADS1299 PWDN | 14 |
| SPI MOSI/MISO/SCK | 11/12/13 |

---

## Signal Processing Pipeline

```
Raw EEG (4000 Hz, 9 channels)
     │
     ▼ Bipolar derivation (CH0 - CH6)
Single channel (4000 Hz)
     │
     ▼ Take every 16th sample (decimation)
250 Hz
     │
     ▼ 6th-order Butterworth (0.5-30 Hz)
250 Hz filtered
     │
     ▼ Resample (5:2 ratio)
100 Hz
     │
     ▼ 30-second window (3000 samples)
     │
     ▼ Z-score normalization
     │
     ▼ CNN inference (~130ms)
     │
Sleep Stage + Confidence
```

---

## SD Card Setup

### Required Files

```
SD Card Root/
├── config.txt                     # Configuration
├── SdioLogger_miklos_night_2.bin  # EEG data (playback mode)
└── data/
    └── reference_predictions.csv  # Validation data (optional)
```

### config.txt Example

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

---

## Project Structure

```
sleep_headband_firmware/
├── src/
│   ├── main_playback_inference.cpp  # Playback mode (default)
│   ├── main_realtime_inference.cpp  # Real-time mode
│   ├── main_data_streaming.cpp      # Debug mode
│   ├── EEGProcessor.cpp
│   ├── MLInference.cpp
│   ├── PreprocessingPipeline.cpp
│   └── ...
├── include/
│   ├── model.h                      # TFLite model
│   ├── ADS1299_Custom.h
│   ├── EEGProcessor.h
│   ├── MLInference.h
│   ├── PreprocessingPipeline.h
│   └── ...
├── docs/                            # Documentation
│   ├── QUICK_START.md
│   ├── ARCHITECTURE.md
│   ├── HARDWARE_SETUP.md
│   ├── VALIDATION_GUIDE.md
│   ├── TROUBLESHOOTING.md
│   └── API_REFERENCE.md
├── tools/                           # Python validation scripts
├── platformio.ini                   # Build configuration
├── CLAUDE.md                        # AI assistant context
└── README.md                        # This file
```

---

## Build Commands

```bash
# Build project
pio run

# Clean build
pio run --target clean

# Upload to Teensy
pio run --target upload

# Serial monitor
pio device monitor
```

---

## Interactive Commands (Playback Mode)

| Key | Action |
|-----|--------|
| `p` | Toggle serial plotting |
| `i` | Toggle ML inference |
| `s` | Show statistics |
| `d` | Toggle debug logging |
| `r` | Restart playback |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Path too long (Windows) | Move to `C:\Projects\sleep_firmware` |
| SD card fails | Reformat as FAT32 |
| No serial output | Check COM port, baud 115200, reset Teensy |
| Low validation (<70%) | Check TFLite library version |

See [Troubleshooting Guide](docs/TROUBLESHOOTING.md) for detailed solutions.

---

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - AI assistant context and project guidance
- **[PENDING_TFLITE_FIX.md](PENDING_TFLITE_FIX.md)** - Validation status and history

---

## License

TODO: Add license
