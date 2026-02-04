# Hardware Setup Guide

This guide covers hardware configuration for the sleep headband firmware.

---

## Required Components

| Component | Description | Notes |
|-----------|-------------|-------|
| Teensy 4.1 | Main microcontroller | Must have PSRAM chip installed |
| ADS1299 | EEG acquisition IC | 8-channel, 24-bit ADC |
| microSD card | Data storage | FAT32 formatted, Class 10+ |
| USB cable | Programming/serial | Must be data-capable |

---

## Teensy 4.1 Pinout

### ADS1299 Connection

```
Teensy 4.1                    ADS1299
─────────────────────────────────────────
Pin 7  (CS)      ─────────>  CS (Chip Select)
Pin 22 (DRDY)    <─────────  DRDY (Data Ready, interrupt)
Pin 15 (START)   ─────────>  START
Pin 14 (PWDN)    ─────────>  PWDN (Power Down)
Pin 11 (MOSI)    ─────────>  DIN (SPI Data In)
Pin 12 (MISO)    <─────────  DOUT (SPI Data Out)
Pin 13 (SCK)     ─────────>  SCLK (SPI Clock)
GND              ─────────>  DGND
3.3V             ─────────>  DVDD (Digital Supply)
```

### SD Card (Built-in)

The Teensy 4.1 has a built-in SD card slot using SDIO interface. No additional wiring needed.

---

## ADS1299 Configuration

### Default Settings (in firmware)

| Parameter | Value | Register |
|-----------|-------|----------|
| Sample Rate | 4000 Hz (or 1000 Hz) | CONFIG1 (0x02) |
| Gain | 24x | CHnSET (0x60) |
| Input Type | Normal | CHnSET |
| Reference | Internal 4.5V | CONFIG3 |

**Supported sample rates:** 1000 Hz or 4000 Hz (configured in `config.txt`)

### Modifying Pin Assignments

Edit `include/ADS1299_Custom.h`:

```cpp
#define ADS1299_CS1   7    // Chip Select
#define ADS1299_DRDY  22   // Data Ready (interrupt pin)
#define ADS1299_START_PIN 15   // Start conversion pin
#define ADS1299_PWDN  14   // Power down
```

---

## SD Card Setup

### Formatting

1. Use FAT32 filesystem (not exFAT or NTFS)
2. Cluster size: Default (usually 32 KB)
3. Quick format is fine

### Required Files

```
SD Card Root/
├── config.txt                     # Configuration file
├── SdioLogger_miklos_night_2.bin  # EEG data file
└── data/
    └── reference_predictions.csv  # Validation data (optional)
```

### config.txt Example (4kHz, 9 channels)

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

### config.txt Example (1kHz with accelerometer)

```ini
datafile=new_recording.bin
sample_rate=1000
channels=12
format=int32
gain=24
vref=4.5
bipolar_channel_positive=0
bipolar_channel_negative=6
has_accelerometer=true
accel_channel_x=8
accel_channel_y=9
accel_channel_z=10
```

**Note:** Accelerometer data is converted to g units: `g = raw * 16.0 / 4095.0`

---

## Power Requirements

### Teensy 4.1

| Source | Voltage | Current |
|--------|---------|---------|
| USB | 5V | 100-500 mA |
| VIN | 5-12V | 100-500 mA |

### ADS1299

| Rail | Voltage | Current |
|------|---------|---------|
| DVDD | 1.8-3.6V | ~10 mA |
| AVDD | 4.75-5.25V | ~10 mA |

**Total system:** ~200-300 mA typical during operation

---

## First-Time Hardware Test

### Step 1: Test Teensy (No ADS1299)

1. Connect Teensy via USB
2. Build and upload playback mode (default)
3. Insert SD card with test data
4. Open serial monitor at 115200 baud

**Expected:** EEG data playback from SD card

### Step 2: Test ADS1299 Connection

1. Wire ADS1299 to Teensy as shown above
2. Switch to real-time mode in platformio.ini
3. Build and upload
4. Open serial monitor

**Expected:**
```
Initializing ADS1299 EEG acquisition...
ADS1299 initialized successfully
Data acquisition started at 4000Hz
```

### Step 3: Verify Data Quality

1. Connect electrodes (or short inputs for testing)
2. Monitor serial output for data values
3. Check for reasonable signal levels (0-200 µV typical)

---

## Troubleshooting Hardware

### ADS1299 Won't Initialize

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| "Failed to initialize ADS1299" | Wiring error | Check all connections |
| No response | Power issue | Verify 3.3V and 5V rails |
| SPI errors | Clock/data mismatch | Check MOSI/MISO not swapped |

### SD Card Issues

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| "SD card initialization failed" | Wrong format | Reformat as FAT32 |
| "File not found" | Wrong filename | Check spelling, case |
| Slow performance | Card speed | Use Class 10 or faster |

### Signal Quality Issues

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Very large values | Gain too high | Reduce gain in config |
| All zeros | Electrodes disconnected | Check electrode connections |
| 50/60 Hz noise | Line interference | Improve shielding, grounding |

---

## Advanced Configuration

### Multiple ADS1299 Chips

For more than 8 channels, cascade multiple ADS1299 chips:

1. Connect DOUT of first chip to DIN of second
2. Share CS, SCLK, START, PWDN
3. Use daisy-chain mode (DAISY_IN/OUT pins)
4. Modify firmware to read 16+ channels

### External Power Supply

For battery operation:

1. Use 3.7V LiPo with boost converter to 5V
2. Connect to Teensy VIN pin
3. Add voltage monitoring for battery level
4. Expected runtime: ~4-6 hours with 2000 mAh battery

---

## Safety Notes

- EEG equipment is not medical grade without proper certification
- Do not use during electrical storms
- Ensure proper electrical isolation for human subjects
- Follow your institution's ethics and safety protocols
