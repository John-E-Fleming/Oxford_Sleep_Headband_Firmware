# API Reference

Technical reference for key classes and functions in the sleep headband firmware.

---

## Core Classes

### PreprocessingPipeline (Default)

Default signal preprocessing from 4kHz to 100Hz via 250Hz intermediate.

**Header:** `include/PreprocessingPipeline.h`

```cpp
class PreprocessingPipeline {
public:
    PreprocessingPipeline();
    void reset();
    bool processSample(float input_4000hz, float& output_100hz);
};
```

**Pipeline:** 4kHz → 250Hz (decimate) → filter@250Hz → 100Hz

#### Methods

| Method | Description |
|--------|-------------|
| `PreprocessingPipeline()` | Constructor, initializes all buffers |
| `reset()` | Reset internal state (call when restarting) |
| `processSample(input, output)` | Process one 4kHz sample, returns true when 100Hz sample ready |

---

### PreprocessingPipelineAlt (Recommended)

Alternative preprocessing pipelines with better accuracy. **Option D is recommended (89.1% agreement).**

**Header:** `include/PreprocessingPipelineAlt.h`

```cpp
class PreprocessingPipelineAlt {
public:
    PreprocessingPipelineAlt();
    void reset();
    bool processSample(float input, float& output_100hz);
};
```

**Pipeline options (selected via build flags):**

| Build Flag | Pipeline | Agreement |
|------------|----------|-----------|
| `-DUSE_ALT_PREPROCESSING_A` | 4kHz→500Hz(decimate)→100Hz(avg)→filter@100Hz | 86.4% |
| `-DUSE_ALT_PREPROCESSING_B` | 4kHz→500Hz(average)→100Hz(avg)→filter@100Hz | 88.4% |
| `-DUSE_ALT_PREPROCESSING_C` | 4kHz→100Hz(decimate)→filter@100Hz | 73.6% |
| `-DUSE_ALT_PREPROCESSING_D` | 4kHz→100Hz(average)→filter@100Hz | **89.1%** |

#### Usage Example

```cpp
#ifdef USE_ALT_PREPROCESSING_D
PreprocessingPipelineAlt pipeline;  // Uses Option D
#else
PreprocessingPipeline pipeline;      // Uses Default
#endif

float input_4khz = ads.convertToMicrovolts(rawData[0]) - ads.convertToMicrovolts(rawData[6]);
float output_100hz;

if (pipeline.processSample(input_4khz, output_100hz)) {
    // New 100Hz sample is ready
    eegProcessor.addFilteredSample(output_100hz);
}
```

---

### EEGProcessor

Handles windowing, buffering, and normalization for ML inference.

**Header:** `include/EEGProcessor.h`

```cpp
class EEGProcessor {
public:
    EEGProcessor();
    bool begin();
    void configureSlidingWindow(int window_seconds, int interval_seconds);
    void addSample(float* channels);          // Multi-channel (legacy)
    void addFilteredSample(float sample);     // Single-channel (preferred)
    bool isWindowReady();
    bool isInferenceTimeReady();
    bool getProcessedWindow(float* output_buffer);
    void markInferenceComplete();
    float getFilteredMean() const;
    float getFilteredStd() const;
};
```

#### Key Methods

| Method | Description |
|--------|-------------|
| `begin()` | Initialize processor, returns false on failure |
| `configureSlidingWindow(window, interval)` | Set window size and slide interval in seconds |
| `addFilteredSample(sample)` | Add one preprocessed 100Hz sample |
| `isInferenceTimeReady()` | Check if enough samples for inference |
| `getProcessedWindow(output)` | Get Z-score normalized window (3000 floats) |
| `markInferenceComplete()` | Reset timer for sliding window |

#### Usage Example

```cpp
EEGProcessor processor;
processor.begin();
processor.configureSlidingWindow(30, 30);  // 30s window, no overlap

// In loop:
processor.addFilteredSample(sample_100hz);

if (processor.isInferenceTimeReady()) {
    float window[3000];
    if (processor.getProcessedWindow(window)) {
        // Run inference on window
        mlInference.predict(window, output, epoch_count);
        processor.markInferenceComplete();
    }
}
```

---

### MLInference

TensorFlow Lite Micro wrapper for sleep stage classification.

**Header:** `include/MLInference.h`

```cpp
class MLInference {
public:
    MLInference();
    ~MLInference();
    bool begin(bool use_dummy = false);
    bool predict(float* input_data, float* output_data, int epoch_index);
    SleepStage getPredictedStage(float* output_data);
    int getInputSize() const;
    int getOutputSize() const;
    unsigned long getLastInferenceTime() const;
    bool isUsingRealModel() const;
    bool getInputQuantizationParams(float& scale, int32_t& zero_point) const;
};
```

#### Key Methods

| Method | Description |
|--------|-------------|
| `begin(use_dummy)` | Initialize TFLite interpreter |
| `predict(input, output, epoch)` | Run inference, returns false on error |
| `getPredictedStage(output)` | Convert probabilities to SleepStage enum |
| `getLastInferenceTime()` | Timing in microseconds |

#### Usage Example

```cpp
MLInference ml;
ml.begin();

float input[3001];   // 3000 samples + epoch index placeholder
float output[5];     // 5 class probabilities

if (ml.predict(input, output, epoch_index)) {
    SleepStage stage = ml.getPredictedStage(output);
    float confidence = *std::max_element(output, output + 5);
}
```

---

### ADS1299_Custom

Hardware interface for ADS1299 EEG acquisition IC.

**Header:** `include/ADS1299_Custom.h`

```cpp
class ADS1299_Custom {
public:
    ADS1299_Custom();
    bool begin();
    void startAcquisition();
    void stopAcquisition();
    bool dataReady();
    void readChannelData(int32_t* channelData);
    bool getLatestData(int32_t* channelData);
    float convertToMicrovolts(int32_t adcValue);
    void setGain(uint8_t gain);
    void setSampleRate(uint8_t sampleRate);
    static void dataReadyISR();
};
```

#### Key Methods

| Method | Description |
|--------|-------------|
| `begin()` | Initialize SPI, configure ADS1299 |
| `startAcquisition()` | Begin continuous data mode |
| `getLatestData(data)` | Get most recent sample from ISR buffer |
| `convertToMicrovolts(adc)` | Convert raw 24-bit value to µV |

#### Usage Example

```cpp
ADS1299_Custom ads;
ads.begin();
ads.startAcquisition();

int32_t rawData[8];
if (ads.getLatestData(rawData)) {
    for (int i = 0; i < 8; i++) {
        float uv = ads.convertToMicrovolts(rawData[i]);
    }
}
```

---

### EEGFileReader

SD card binary file reader for playback mode.

**Header:** `include/EEGFileReader.h`

```cpp
class EEGFileReader {
public:
    EEGFileReader();
    bool begin(const char* filename);
    bool readNextSample(float* channels);
    void setFormat(EEGDataFormat format, int numChannels);
    void seekToTime(float seconds);
    uint32_t getFileSize();
    float getDurationSeconds();
};
```

#### Key Methods

| Method | Description |
|--------|-------------|
| `begin(filename)` | Open file on SD card |
| `readNextSample(channels)` | Read next sample, returns false at EOF |
| `setFormat(format, channels)` | Configure data format |
| `seekToTime(seconds)` | Jump to specific time in file |

---

## Enumerations

### SleepStage

```cpp
enum SleepStage {
    WAKE = 0,           // yy0 - Awake
    N1_VERY_LIGHT = 1,  // yy1 - Stage N1
    N2_LIGHT_SLEEP = 2, // yy2 - Stage N2
    N3_DEEP_SLEEP = 3,  // yy3 - Stage N3 (SWS)
    REM_SLEEP = 4       // yy4 - REM sleep
};
```

### EEGDataFormat

```cpp
enum EEGDataFormat {
    FORMAT_INT32,   // 32-bit signed integers
    FORMAT_INT16,   // 16-bit signed integers
    FORMAT_FLOAT32  // 32-bit floats
};
```

---

## Configuration Constants

### model.h

```cpp
#define MODEL_INPUT_SIZE 3001     // 3000 samples + 1 epoch index
#define MODEL_OUTPUT_SIZE 5       // 5 sleep stages
#define TENSOR_ARENA_SIZE 163840  // 160 KB for TFLite
```

### EEGProcessor.h

```cpp
#define ADS1299_CHANNELS 9        // Number of input channels
#define ML_SAMPLE_RATE 100        // Processing rate (Hz)
#define ML_WINDOW_SIZE_SECONDS 30 // Window duration
#define ML_WINDOW_SIZE_SAMPLES 3000 // Samples per window
```

### ADS1299_Custom.h

```cpp
#define ADS1299_CS1   7    // Chip Select pin
#define ADS1299_DRDY  22   // Data Ready interrupt pin
#define ADS1299_START_PIN 15   // Start pin
#define ADS1299_PWDN  14   // Power down pin
#define ADS1299_CHANNELS 8 // Hardware channels
```

---

## Configuration File (config.txt)

### Parameters

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `datafile` | string | Binary data filename | (required) |
| `sample_rate` | int | Input sample rate (Hz) | 4000 |
| `channels` | int | Number of channels | 9 |
| `format` | string | Data format (int32/int16/float32) | int32 |
| `gain` | int | ADC gain setting | 24 |
| `vref` | float | Reference voltage | 4.5 |
| `bipolar_channel_positive` | int | Positive electrode index | 0 |
| `bipolar_channel_negative` | int | Negative electrode index | 6 |
| `has_accelerometer` | bool | Whether file has accelerometer data | false |
| `accel_channel_x` | int | Accelerometer X channel index | 8 |
| `accel_channel_y` | int | Accelerometer Y channel index | 9 |
| `accel_channel_z` | int | Accelerometer Z channel index | 10 |

### Example (4kHz, 9 channels)

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

### Example (1kHz with accelerometer)

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

**Note:** Accelerometer data is converted to g units using: `g = raw * 16.0 / 4095.0`

---

## Adding New Features

### Adding a New Sleep Stage

1. Add to `SleepStage` enum in `model.h`
2. Update `MODEL_OUTPUT_SIZE`
3. Retrain model with new class
4. Update stage string conversion in main files

### Changing Model Architecture

1. Export new model as `.tflite`
2. Convert to C array: `xxd -i model.tflite > model.h`
3. Update `MODEL_INPUT_SIZE` and `MODEL_OUTPUT_SIZE`
4. Adjust `TENSOR_ARENA_SIZE` if needed
5. Rebuild and validate

### Adding New Preprocessing Stage

1. Create new class in `include/` and `src/`
2. Add to pipeline in `PreprocessingPipeline.cpp` or `PreprocessingPipelineAlt.cpp`
3. Update reset() to clear new state
4. Add build flag to `platformio.ini` if creating a new option
5. Validate against Python reference using:
   ```bash
   python tools/run_inference.py data.bin --output predictions.csv
   python tools/compare_preprocessing_options.py
   ```

### Selecting Preprocessing Pipeline

Add one of these flags to `build_flags` in `platformio.ini`:

```ini
; Default (no flag): 4kHz→250Hz→filter@250Hz→100Hz (81.4% agreement)
-DUSE_ALT_PREPROCESSING_A  ; Option A: 86.4% agreement
-DUSE_ALT_PREPROCESSING_B  ; Option B: 88.4% agreement
-DUSE_ALT_PREPROCESSING_C  ; Option C: 73.6% agreement (not recommended)
-DUSE_ALT_PREPROCESSING_D  ; Option D: 89.1% agreement (RECOMMENDED)
```
