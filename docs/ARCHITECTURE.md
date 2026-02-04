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

### Input Stage (1000 Hz or 4000 Hz)

Data can come from two sources:

1. **ADS1299 Hardware** (`main_realtime_inference.cpp`)
   - Live EEG acquisition at 4000 Hz
   - 8 channels via SPI interface
   - Interrupt-driven data collection

2. **SD Card Playback** (`main_playback_inference.cpp`)
   - Pre-recorded EEG files
   - Binary format (int32, interleaved)
   - Configurable sample rate (1kHz, 4kHz) via `config.txt`
   - Optional accelerometer channels

### Bipolar Derivation

Raw multi-channel data is converted to a single bipolar signal:

```
bipolar_sample = channel[positive] - channel[negative]
                      (CH0/F3)           (CH6/M2)
```

This matches standard sleep EEG montage (F3-M2).

### Preprocessing Pipeline (Input → 100Hz)

Multiple preprocessing pipelines are available, selected via build flags in `platformio.ini`.

**Option D (Recommended - 89.1% agreement):**
```
4000 Hz input (or 1000 Hz)
     │
     ▼ Average every N samples (N=40 for 4kHz, N=10 for 1kHz)
100 Hz
     │
     ▼ 5th-order Butterworth bandpass (0.5-30 Hz)
100 Hz output
```

**Default (Legacy - 81.4% agreement):**
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

**Preprocessing Pipeline Options:**

| Option | Build Flag | Pipeline | Agreement |
|--------|------------|----------|-----------|
| Default | (none) | 4kHz→250Hz→filter@250Hz→100Hz | 81.4% |
| A | `-DUSE_ALT_PREPROCESSING_A` | 4kHz→500Hz→100Hz→filter@100Hz | 86.4% |
| B | `-DUSE_ALT_PREPROCESSING_B` | 4kHz→500Hz(avg)→100Hz→filter@100Hz | 88.4% |
| C | `-DUSE_ALT_PREPROCESSING_C` | 4kHz→100Hz(decimate)→filter@100Hz | 73.6% |
| **D** | `-DUSE_ALT_PREPROCESSING_D` | **4kHz→100Hz(avg)→filter@100Hz** | **89.1%** |

**Key parameters:**
- Filter: 5th-order Butterworth, 0.5-30 Hz (Option D) or 6th-order (Default)
- Coefficients match Python training pipeline
- Output rate: 100 Hz (model input requirement)
- Supports configurable input sample rates (1kHz, 4kHz)

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
| PreprocessingPipeline | `PreprocessingPipeline.h/.cpp` | Default downsampling, filtering, resampling |
| PreprocessingPipelineAlt | `PreprocessingPipelineAlt.h/.cpp` | Alternative pipelines (Options A-D) |
| TrainingBandpassFilter | `TrainingBandpassFilter.h/.cpp` | Butterworth filter (6th-order, 250Hz) |
| BandpassFilter100Hz | `BandpassFilter100Hz.h/.cpp` | Butterworth filter (5th-order, 100Hz) |
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
│   ├── PreprocessingPipeline.cpp    # Default pipeline (250Hz filter)
│   ├── PreprocessingPipelineAlt.cpp # Alt pipelines (Options A-D)
│   ├── BandpassFilter100Hz.cpp      # 100Hz bandpass filter
│   └── ...
├── include/                          # Header files
│   ├── model.h                      # TFLite model (byte array)
│   ├── ADS1299_Custom.h             # Hardware interface
│   ├── EEGProcessor.h               # Processor interface
│   ├── MLInference.h                # ML interface
│   ├── PreprocessingPipeline.h      # Default pipeline interface
│   ├── PreprocessingPipelineAlt.h   # Alt pipelines interface
│   ├── BandpassFilter100Hz.h        # 100Hz filter interface
│   └── ...
├── docs/                            # Documentation
├── tools/                           # Python tools
│   ├── run_inference.py             # Flexible Python inference
│   ├── visualize_session.py         # Session visualization
│   ├── compare_preprocessing_options.py  # Pipeline comparison
│   └── ...
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
| Input sample rate | 1000 Hz or 4000 Hz (configurable) |
| Output sample rate | 100 Hz |
| Window size | 30 seconds (3000 samples) |
| Inference time | ~130 ms |
| Real-time factor | ~90x faster than real-time |
| Validation agreement (Option D) | 89.1% |

---

## Validation Status

Multiple preprocessing pipelines have been validated against Python reference:

| Pipeline | Python Agreement | Teensy Agreement | Notes |
|----------|-----------------|------------------|-------|
| Default | 86.8% | 81.4% | Original implementation |
| Option A | 86.4% | ~86% | Decimate to 500Hz first |
| Option B | 88.4% | ~88% | Average to 500Hz first |
| Option C | 73.6% | ~74% | Direct decimate (not recommended) |
| **Option D** | **89.1%** | **~89%** | **Recommended** |

**Recommendation:** Use Option D (`-DUSE_ALT_PREPROCESSING_D` in platformio.ini) for best accuracy.

See [VALIDATION_GUIDE.md](VALIDATION_GUIDE.md) for details.
