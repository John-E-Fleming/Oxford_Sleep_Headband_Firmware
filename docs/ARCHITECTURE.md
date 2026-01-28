# System Architecture

This document describes the technical architecture of the sleep headband firmware.

---

## System Overview

The firmware implements a complete EEG signal processing pipeline for real-time sleep stage classification on the Teensy 4.1 microcontroller.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SLEEP HEADBAND SYSTEM                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌──────────────────────────────────────────────┐   │
│  │  ADS1299    │    │           PreprocessingPipeline               │   │
│  │  (4kHz)     │───>│  4kHz → 250Hz → Butterworth → 100Hz          │   │
│  │  or SD Card │    │                                               │   │
│  └─────────────┘    └──────────────────────────────────────────────┘   │
│                                          │                              │
│                                          ▼                              │
│                     ┌──────────────────────────────────────────────┐   │
│                     │              EEGProcessor                     │   │
│                     │  30-second window → Z-score normalization    │   │
│                     │  Circular buffer → Sliding window            │   │
│                     └──────────────────────────────────────────────┘   │
│                                          │                              │
│                                          ▼                              │
│                     ┌──────────────────────────────────────────────┐   │
│                     │              MLInference                      │   │
│                     │  TensorFlow Lite Micro → CNN → 5 classes     │   │
│                     │  Input: 3000 samples + epoch index           │   │
│                     └──────────────────────────────────────────────┘   │
│                                          │                              │
│                                          ▼                              │
│                     ┌──────────────────────────────────────────────┐   │
│                     │              Output                           │   │
│                     │  Sleep Stage: Wake / N1 / N2 / N3 / REM      │   │
│                     │  Confidence: 0.0 - 1.0                        │   │
│                     └──────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Input Stage (4000 Hz)

Data can come from two sources:

1. **ADS1299 Hardware** (`main_realtime_inference.cpp`)
   - Live EEG acquisition at 4000 Hz
   - 8 channels via SPI interface
   - Interrupt-driven data collection

2. **SD Card Playback** (`main_playback_inference.cpp`)
   - Pre-recorded EEG files
   - Binary format (int32, interleaved)
   - Configurable via `config.txt`

### Bipolar Derivation

Raw multi-channel data is converted to a single bipolar signal:

```
bipolar_sample = channel[positive] - channel[negative]
                      (CH0/F3)           (CH6/M2)
```

This matches standard sleep EEG montage (F3-M2).

### Preprocessing Pipeline (4kHz → 100Hz)

The `PreprocessingPipeline` class performs three stages:

```
4000 Hz input
     │
     ▼ Take every 16th sample (decimation)
250 Hz
     │
     ▼ 6th-order Butterworth bandpass (0.5-30 Hz)
250 Hz filtered
     │
     ▼ Resample 5:2 ratio (with interpolation)
100 Hz output
```

**Key parameters:**
- Filter: 6th-order Butterworth, 0.5-30 Hz
- Coefficients match Python training pipeline exactly
- Output rate: 100 Hz (model input requirement)

### EEG Processor (Windowing)

The `EEGProcessor` class handles:

1. **Circular buffer** - Stores 100 Hz samples
2. **30-second window** - 3000 samples per inference
3. **Z-score normalization** - Per-window standardization
4. **Sliding window** - Configurable overlap (default: 0%)

### ML Inference (TensorFlow Lite Micro)

The `MLInference` class runs the CNN model:

**Model architecture:**
- Input 1: EEG data (shape: 1, 1, 3000, 1)
- Input 2: Epoch index (shape: 1, 1)
- Output: 5 probabilities (Wake, N1, N2, N3, REM)

**Model format:**
- TensorFlow Lite (FLOAT32)
- Tensor arena: 160 KB in internal RAM
- Inference time: ~130 ms per epoch

---

## Component Responsibilities

| Component | File | Responsibility |
|-----------|------|----------------|
| ADS1299_Custom | `ADS1299_Custom.h/.cpp` | Hardware SPI interface, ADC conversion |
| PreprocessingPipeline | `PreprocessingPipeline.h/.cpp` | Downsampling, filtering, resampling |
| TrainingBandpassFilter | `TrainingBandpassFilter.h/.cpp` | Butterworth filter coefficients |
| EEGProcessor | `EEGProcessor.h/.cpp` | Windowing, normalization, buffering |
| MLInference | `MLInference.h/.cpp` | TFLite interpreter, model execution |
| EEGFileReader | `EEGFileReader.h/.cpp` | SD card file reading |
| Config | `Config.h/.cpp` | Configuration file parsing |

---

## File Organization

```
sleep_headband_firmware/
├── src/                              # Source files
│   ├── main_realtime_inference.cpp  # Real-time mode (ADS1299)
│   ├── main_playback_inference.cpp  # Playback mode (SD card)
│   ├── main_data_streaming.cpp      # Debug mode (no ML)
│   ├── EEGProcessor.cpp             # Signal processing
│   ├── MLInference.cpp              # ML inference
│   ├── PreprocessingPipeline.cpp    # Pipeline implementation
│   └── ...
├── include/                          # Header files
│   ├── model.h                      # TFLite model (byte array)
│   ├── ADS1299_Custom.h             # Hardware interface
│   ├── EEGProcessor.h               # Processor interface
│   ├── MLInference.h                # ML interface
│   ├── PreprocessingPipeline.h      # Pipeline interface
│   └── ...
├── docs/                            # Documentation
├── tools/                           # Python validation tools
├── data/                            # Sample data files
└── platformio.ini                   # Build configuration
```

---

## Operational Modes

The firmware supports three operational modes, selected via `platformio.ini`:

### 1. Playback Mode (Default)

**File:** `main_playback_inference.cpp`

**Purpose:** Validation and testing with recorded data

**Features:**
- Reads EEG from SD card binary files
- Full preprocessing pipeline
- ML inference every 30 seconds
- Optional validation against reference predictions
- Interactive serial commands

**Use when:** Testing, validation, debugging

### 2. Real-Time Mode

**File:** `main_realtime_inference.cpp`

**Purpose:** Production use with live EEG

**Features:**
- Live ADS1299 data acquisition
- Same preprocessing pipeline as playback
- Real-time sleep stage output
- Statistics reporting

**Use when:** Actual sleep studies

### 3. Data Streaming Mode

**File:** `main_data_streaming.cpp`

**Purpose:** Hardware debugging

**Features:**
- Simple SD card streaming
- No ML inference
- Raw data output

**Use when:** Testing SD card, debugging hardware

---

## Build System

### platformio.ini Structure

```ini
[env:teensy41]
platform = teensy
board = teensy41
framework = arduino

lib_deps =
    wizard97/RingBuf@^2.0.1
    arduino-libraries/SD@^1.3.0
    https://github.com/tensorflow/tflite-micro-arduino-examples.git#2be8092d

build_flags =
    -DENABLE_VALIDATION_MODE  # Enable reference comparison

build_src_filter = +<*> -<main_realtime_inference.cpp> -<main_data_streaming.cpp> ...
```

### Switching Modes

Edit `platformio.ini` and comment/uncomment the appropriate `build_src_filter` line:

```ini
; PLAYBACK MODE (DEFAULT)
build_src_filter = +<*> -<main_realtime_inference.cpp> -<main_data_streaming.cpp> ...

; REAL-TIME MODE
; build_src_filter = +<*> -<main_playback_inference.cpp> -<main_data_streaming.cpp> ...
```

---

## Memory Layout

### Teensy 4.1 Memory

| Region | Size | Usage |
|--------|------|-------|
| FLASH | 8 MB | Code, model, constants |
| RAM1 | 512 KB | Variables, code, stack |
| RAM2 | 512 KB | Heap, tensor arena |
| PSRAM | 8 MB | Optional external storage |

### Key Allocations

| Component | Size | Location |
|-----------|------|----------|
| Tensor arena | 160 KB | RAM2 (internal) |
| Circular buffer | 12 KB | RAM1 |
| Model weights | ~80 KB | FLASH |

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Input sample rate | 4000 Hz |
| Output sample rate | 100 Hz |
| Window size | 30 seconds (3000 samples) |
| Inference time | ~130 ms |
| Real-time factor | ~90x faster than real-time |
| Validation agreement | 81.4% |

---

## Validation Status

The preprocessing pipeline has been validated against Python reference:

| Stage | Agreement | Notes |
|-------|-----------|-------|
| Python (reference EEG) | 100% | Model is correct |
| Python (our preprocessing) | 86.8% | Preprocessing verified |
| Teensy (embedded) | 81.4% | Production ready |

See [VALIDATION_GUIDE.md](VALIDATION_GUIDE.md) for details.
