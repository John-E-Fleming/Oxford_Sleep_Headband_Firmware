# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PlatformIO project for a sleep headband firmware running on Teensy 4.1, designed to:
- Interface with ADS1299 EEG acquisition chips to capture brain signals
- Stream EEG data from SD card files for testing/playback
- Process EEG data using TinyML for real-time sleep stage classification
- Support multiple operational modes (live acquisition, file playback, ML inference)

The firmware implements a complete EEG signal processing pipeline from raw ADC data to sleep stage predictions using machine learning models optimized for microcontrollers.

## Build Commands

```bash
# Build the project
pio run

# Upload to Teensy 4.1
pio run --target upload

# Open serial monitor
pio device monitor

# Clean build artifacts
pio run --target clean
```

## Build Configuration

The project uses `build_src_filter` in `platformio.ini` to select which main file to compile:

- **File playback mode (default)**: `build_src_filter = +<*> -<main.cpp> -<main_ml.cpp> -<simple_test.cpp>`
  - Uses `test_eeg_playback.cpp` as main
  - Streams EEG data from SD card files
  - Supports ML inference on recorded data

- **Live ADS1299 mode**: Uncomment `build_src_filter = +<*> -<main.cpp> -<test_eeg_playback.cpp>`
  - Uses `main_ml.cpp` as main  
  - Real-time EEG acquisition from ADS1299
  - Live ML inference every 4 seconds

- **SD streaming mode**: Uncomment `build_src_filter = +<*> -<main_ml.cpp> -<test_eeg_playback.cpp>`
  - Uses `main.cpp` as main
  - Basic file streaming functionality

## Architecture Overview

### Core Components

1. **ADS1299_Custom** (`ADS1299_Custom.h/.cpp`) - Hardware interface for EEG acquisition
   - SPI communication with ADS1299 chips
   - Interrupt-driven data collection
   - Configurable gain (1x-24x) and sample rates (250Hz-16kHz)
   - Raw ADC to microvolt conversion

2. **EEGProcessor** (`EEGProcessor.h/.cpp`) - Signal processing pipeline
   - Circular buffering for continuous data streams
   - Preprocessing: filtering, normalization, windowing
   - Memory-optimized for Teensy 4.1 constraints
   - 1-second processing windows (4000 samples at 4kHz)

3. **MLInference** (`MLInference.h/.cpp`) - TinyML sleep stage classification
   - TensorFlow Lite Micro integration
   - 4-class sleep stage output: Wake, Light Sleep, Deep Sleep, REM
   - Optimized for 10KB tensor arena size

4. **FileStreamer/EEGFileReader** - SD card data handling
   - Supports multiple EEG file formats (int32, binary)
   - 4000Hz playback timing for realistic simulation
   - Configuration via `config.txt`

### Data Flow

```
SD Card/ADS1299 → EEGProcessor → MLInference → Serial Output
     ↓               ↓              ↓
   Raw Data    → Windowing  →  Sleep Stage
   (int32)       (4000 samples)  Classification
```

### Configuration

- **Hardware config**: Pin assignments in `ADS1299_Custom.h` 
- **Data config**: `data/config.txt` specifies file paths and acquisition parameters
- **ML config**: Model parameters in `model.h` (input size, tensor arena)

### Memory Management

The firmware is optimized for Teensy 4.1's memory constraints:
- Circular buffers instead of large arrays  
- Reduced window sizes (1 second vs longer)
- Configurable tensor arena size (10KB default)
- Static memory allocation where possible

## Development Notes

### Testing Modes

The firmware supports interactive commands in file playback mode:
- `p` - Toggle serial plotting  
- `i` - Toggle ML inference
- `s` - Show statistics
- `r` - Restart playback

### SD Card Setup

Place EEG data files on SD card and configure `data/config.txt`:
```
datafile=SdioLogger_miklos_night_2.bin
sample_rate=4000
channels=9
format=int32
gain=24
vref=4.5
```

### Pin Configuration

Default ADS1299 pin assignments (modify in `ADS1299_Custom.h`):
- CS: Pin 7
- DRDY: Pin 22 (interrupt)
- START: Pin 15
- PWDN: Pin 14

### Serial Output Format

The firmware outputs CSV data for analysis:
`Time,CH1,CH2,CH3,CH4,CH5,CH6,CH7,CH8,CH9[,Stage,Confidence]`