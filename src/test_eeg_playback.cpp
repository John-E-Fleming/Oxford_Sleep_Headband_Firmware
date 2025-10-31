#include <Arduino.h>
#include <SPI.h>
#include <SdFat.h>
#include "EEGFileReader.h"
#include "EEGProcessor.h"
#include "MLInference.h"
#include "model.h"
#include "Config.h"
#include "PreprocessingPipeline.h"
#include "EEGQualityChecker.h"
#include "InferenceLogger.h"
#include "DebugLogger.h"

// SdFat object for EEG file reading (matching colleague's setup)
SdFat sd;

// Configuration
Config config;
unsigned long SAMPLE_INTERVAL_US = 250; // Will be set from config (4000Hz = 250us)

// EEG data components
EEGFileReader eegReader;
EEGProcessor eegProcessor;
MLInference mlInference;
PreprocessingPipeline preprocessingPipeline;
EEGQualityChecker qualityChecker;
InferenceLogger inferenceLogger;
DebugLogger debugLogger;

// Playback speed control (MUST BE DECLARED BEFORE USAGE BELOW)
const bool FAST_PLAYBACK = true;  // Set to true to process as fast as possible (no timing delays)
const float PLAYBACK_SPEED_MULTIPLIER = 100.0f; // Speed multiplier if not using FAST_PLAYBACK (e.g., 10x = 10x faster)

// Data buffers
float eeg_sample[ADS1299_CHANNELS];
float processed_window[MODEL_INPUT_SIZE];
float ml_output[MODEL_OUTPUT_SIZE];
int8_t quantized_input[MODEL_INPUT_SIZE];  // For debug logging of quantized data

// Circular buffer for storing 100Hz samples for debug logging (30 seconds = 3000 samples)
#define DEBUG_BUFFER_SIZE 3000
float debug_100hz_buffer[DEBUG_BUFFER_SIZE];
int debug_100hz_index = 0;
bool debug_buffer_full = false;

unsigned long last_sample_time = 0;

// Statistics and control
unsigned long sample_count = 0;
unsigned long inference_count = 0;
bool enable_inference = true;
bool enable_serial_plot = !FAST_PLAYBACK;  // Disable plotting in fast mode (printing slows down processing)
bool enable_quality_check = true;
bool enable_inference_logging = true;

// Processing variables
unsigned long processed_count = 0;

// Test parameters
const float START_TIME_SECONDS = 0.0f;    // Start from beginning
const float MAX_DURATION_SECONDS = 300.0f; // Test for 5 minutes max

// Sliding window configuration
const int INFERENCE_WINDOW_SECONDS = 30;   // 30-second analysis window
const int INFERENCE_INTERVAL_SECONDS = 30; // Perform inference every 30 seconds (non-overlapping)
const float WINDOW_OVERLAP_PERCENT = 100.0f * (INFERENCE_WINDOW_SECONDS - INFERENCE_INTERVAL_SECONDS) / INFERENCE_WINDOW_SECONDS;

// Forward declarations
void handleSerialCommands();
void printStatistics();

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 5000) {} // Wait up to 5 seconds for serial
  
  Serial.println("=== EEG File Playback Test ===");
  Serial.println("Commands:");
  Serial.println("  p - Toggle serial plotting");
  Serial.println("  i - Toggle ML inference");
  Serial.println("  s - Show statistics");
  Serial.println("  q - Toggle quality checking");
  Serial.println("  l - Toggle inference logging");
  Serial.println("  d - Toggle debug logging (CSV output)");
  Serial.println("  x - Export log to SD card");
  Serial.println("  r - Restart playback");
  Serial.println();
  
  // Initialize SD card using SdFat library (matching colleague's setup)
  Serial.println("Initializing SD card...");
  bool sd_ok = false;
  
  // Method 1: SdioConfig with FIFO (exactly like colleague's code)
  if (sd.begin(SdioConfig(FIFO_SDIO))) {
    sd_ok = true;
    Serial.println("SD initialized with SdioConfig(FIFO_SDIO)");
  }
  // Method 2: Fallback to DMA SDIO
  else if (sd.begin(SdioConfig(DMA_SDIO))) {
    sd_ok = true;
    Serial.println("SD initialized with SdioConfig(DMA_SDIO)");
  }
  // Method 3: SPI fallback
  else if (sd.begin(SdSpiConfig(10, DEDICATED_SPI, SD_SCK_MHZ(50)))) {
    sd_ok = true;
    Serial.println("SD initialized with SPI interface");
  }
  
  if (!sd_ok) {
    Serial.println("SD card initialization failed - continuing without SD card for testing");
    Serial.println("Insert SD card with EEG file to enable file playback");
    Serial.println("Make sure SD card is properly inserted in Teensy 4.1 slot");
    enable_inference = false; // Disable since we can't read data
  } else {
    Serial.println("SD card initialized");
    
    // Load configuration from SD card
    Serial.println("Loading configuration...");
    SdFile rootDir;
    if (rootDir.open("/") && loadConfig(rootDir, config)) {
      Serial.println("Configuration loaded successfully:");
      Serial.print("  Datafile: ");
      Serial.println(config.datafile);
      Serial.print("  Sample rate: ");
      Serial.print(config.sample_rate);
      Serial.println(" Hz (NOTE: should be 4000Hz for new preprocessing pipeline)");
      Serial.print("  Bipolar channels: ");
      Serial.print(config.bipolar_channel_positive);
      Serial.print(" - ");
      Serial.println(config.bipolar_channel_negative);

      // Show playback speed info
      if (FAST_PLAYBACK) {
        Serial.println("  Playback mode: FAST (processing as quickly as possible)");
      } else {
        Serial.print("  Playback speed: ");
        Serial.print(PLAYBACK_SPEED_MULTIPLIER, 1);
        Serial.println("x real-time");
      }

      // Set sample interval based on config
      SAMPLE_INTERVAL_US = 1000000 / config.sample_rate; // Convert Hz to microseconds

      // Warn if sample rate is not 4000Hz
      if (config.sample_rate != 4000) {
        Serial.println("WARNING: Preprocessing pipeline expects 4000Hz input!");
        Serial.println("         Please update config.txt sample_rate to 4000");
      }
      
      rootDir.close();
    } else {
      Serial.println("Failed to load config.txt, using defaults");
      config.datafile = "SdioLogger_miklos_night_2.bin";
      config.sample_rate = 4000;
      config.channels = 9;
      config.bipolar_channel_positive = 0;
      config.bipolar_channel_negative = 6;
      SAMPLE_INTERVAL_US = 250; // 4000Hz = 250us per sample
    }
    
    // Open EEG file using config
    Serial.println("Opening EEG file...");
    if (!eegReader.begin(config.datafile)) {
      enable_inference = false;
    } else {
      Serial.println("EEG file opened successfully");
    }
  }
  
  // Set format based on config
  EEGDataFormat format = FORMAT_INT32;
  if (config.format == "float32") format = FORMAT_FLOAT32;
  else if (config.format == "int16") format = FORMAT_INT16;
  eegReader.setFormat(format, config.channels);
  
  // Seek to start time if specified
  if (START_TIME_SECONDS > 0) {
    Serial.print("Seeking to ");
    Serial.print(START_TIME_SECONDS);
    Serial.println(" seconds...");
    eegReader.seekToTime(START_TIME_SECONDS);
  }
  
  // Initialize EEG processor with sliding window configuration
  if (!eegProcessor.begin()) {
    Serial.println("Failed to initialize EEG processor");
    while (1);
  }
  
  // Configure sliding window parameters
  eegProcessor.configureSlidingWindow(INFERENCE_WINDOW_SECONDS, INFERENCE_INTERVAL_SECONDS);
  
  // Initialize inference logger
  if (enable_inference_logging) {
    inferenceLogger.begin(1000); // Pre-allocate for 1000 records
  }

  // Initialize debug logger
  if (debugLogger.begin(&sd)) {
    Serial.println("Debug logger ready (use 'd' to enable)");
  } else {
    Serial.println("Debug logger failed to initialize");
  }

  Serial.print("Sliding window: ");
  Serial.print(INFERENCE_WINDOW_SECONDS);
  Serial.print("s window with ");
  Serial.print(INFERENCE_INTERVAL_SECONDS);
  Serial.print("s slide (");
  Serial.print(WINDOW_OVERLAP_PERCENT, 1);
  Serial.println("% overlap)");
  
  // Initialize ML inference
  Serial.println("Initializing ML inference...");
  bool use_dummy_model = false;  // Using real TensorFlow Lite model
  if (!mlInference.begin(use_dummy_model)) {
    Serial.println("ML inference initialization failed - continuing without ML");
    enable_inference = false;
  } else {
    Serial.print("ML inference ready (");
    Serial.print(mlInference.isUsingRealModel() ? "real model" : "dummy model");
    Serial.println(")");
  }

  Serial.println("Starting EEG playback...");
  
  if (enable_serial_plot) {
    Serial.println("Serial plot format: Time,CH1,CH2,CH3,CH4,CH5,CH6,CH7,CH8,CH9[,Stage,Confidence]");
  }
  
  last_sample_time = micros();
}

void loop() {
  // Handle serial commands (only check every 1000 samples in fast mode to reduce overhead)
  static unsigned long serial_check_counter = 0;
  if (!FAST_PLAYBACK || (serial_check_counter++ % 1000 == 0)) {
    handleSerialCommands();
  }

  // Check if it's time for the next sample
  unsigned long current_time = micros();
  unsigned long sample_interval = FAST_PLAYBACK ? 0 : (SAMPLE_INTERVAL_US / PLAYBACK_SPEED_MULTIPLIER);

  if (current_time - last_sample_time >= sample_interval) {
    
    // Read next sample from file - no synthetic data fallback
    bool has_data = false;
    
    if (sd.exists(config.datafile.c_str())) {
      has_data = eegReader.readNextSample(eeg_sample);
      if (!has_data) {
        Serial.println("End of file reached or read error");
        Serial.print("Total samples processed: ");
        Serial.println(sample_count);
        while(1); // Stop execution
      }
    } else {
      Serial.print("ERROR: EEG file '");
      Serial.print(config.datafile);
      Serial.println("' not found on SD card");
      Serial.println("Available files:");
      // List files on SD card for debugging
      SdFile root;
      if (root.open("/")) {
        while (true) {
          SdFile entry;
          if (!entry.openNext(&root, O_RDONLY)) break;
          char name[64];
          entry.getName(name, sizeof(name));
          Serial.print("  ");
          Serial.println(name);
          entry.close();
        }
        root.close();
      }
      while(1); // Stop execution
    }
    
    if (has_data) {
      sample_count++;
      last_sample_time = current_time;

      // Create bipolar derivation (CH0 - CH6) for ML processing
      float bipolar_sample = eeg_sample[config.bipolar_channel_positive] - eeg_sample[config.bipolar_channel_negative];

      // Process through complete pipeline: 4000Hz → 500Hz → BP filter → Notch → 100Hz
      float output_100hz;
      bool sample_ready = preprocessingPipeline.processSample(bipolar_sample, output_100hz);

      if (sample_ready) {
        // New 100Hz sample is ready after full preprocessing
        processed_count++;

        // Store in circular buffer for debug logging
        debug_100hz_buffer[debug_100hz_index] = output_100hz;
        debug_100hz_index++;
        if (debug_100hz_index >= DEBUG_BUFFER_SIZE) {
          debug_100hz_index = 0;
          debug_buffer_full = true;
        }

        // Add processed sample to ML processor buffer
        eegProcessor.addFilteredSample(output_100hz);

        // ML inference with sliding window (if enabled and ready) - MUST BE OUTSIDE SERIAL PLOT CHECK!
        if (enable_inference && eegProcessor.isInferenceTimeReady()) {

            if (eegProcessor.getProcessedWindow(processed_window)) {
              // Note: epoch_index is now passed as separate parameter to predict()

              // DISABLED: Heavy debug logging (preprocessed/normalized/quantized data)
              // These write ~27KB per inference to SD card, making it extremely slow!
              // Uncomment only if you need to debug the preprocessing pipeline specifically.

              // if (debugLogger.isEnabled() && debug_buffer_full) {
              //   float ordered_buffer[DEBUG_BUFFER_SIZE];
              //   for (int i = 0; i < DEBUG_BUFFER_SIZE; i++) {
              //     int buffer_idx = (debug_100hz_index + i) % DEBUG_BUFFER_SIZE;
              //     ordered_buffer[i] = debug_100hz_buffer[buffer_idx];
              //   }
              //   debugLogger.logPreprocessed100Hz(ordered_buffer, DEBUG_BUFFER_SIZE, inference_count);
              // }

              // if (debugLogger.isEnabled()) {
              //   debugLogger.logNormalizedWindow(processed_window, 3000, inference_count);
              //
              //   float scale;
              //   int32_t zero_point;
              //   if (mlInference.getInputQuantizationParams(scale, zero_point)) {
              //     for (int i = 0; i < 3000; i++) {
              //       int32_t quantized = round(processed_window[i] / scale) + zero_point;
              //       if (quantized > 127) quantized = 127;
              //       if (quantized < -128) quantized = -128;
              //       quantized_input[i] = static_cast<int8_t>(quantized);
              //     }
              //     int32_t epoch_quantized = round(static_cast<float>(inference_count) / scale) + zero_point;
              //     if (epoch_quantized > 127) epoch_quantized = 127;
              //     if (epoch_quantized < -128) epoch_quantized = -128;
              //     quantized_input[3000] = static_cast<int8_t>(epoch_quantized);
              //     debugLogger.logQuantizedInput(quantized_input, 3001, inference_count);
              //   }
              // }

              if (mlInference.predict(processed_window, ml_output, inference_count)) {
                SleepStage predicted_stage = mlInference.getPredictedStage(ml_output);

                float max_confidence = 0.0f;
                for (int i = 0; i < MODEL_OUTPUT_SIZE; i++) {
                  if (ml_output[i] > max_confidence) {
                    max_confidence = ml_output[i];
                  }
                }

                // Convert predicted stage to string for debug logging
                const char* stage_str = "UNKNOWN";
                switch (predicted_stage) {
                  case N3_DEEP_SLEEP: stage_str = "N3"; break;    // yy0
                  case N2_LIGHT_SLEEP: stage_str = "N2"; break;   // yy1
                  case N1_VERY_LIGHT: stage_str = "N1"; break;    // yy2
                  case REM_SLEEP: stage_str = "REM"; break;        // yy3
                  case WAKE: stage_str = "WAKE"; break;            // yy4
                }

                // Debug logging: Log model output probabilities
                if (debugLogger.isEnabled()) {
                  float time_s = processed_count / 100.0f;  // Time in seconds at 100Hz
                  debugLogger.logModelOutput(ml_output, MODEL_OUTPUT_SIZE, inference_count, time_s, stage_str);
                }

                inference_count++;
                eegProcessor.markInferenceComplete();  // Reset sliding window timer

                // Progress updates in fast playback mode (every 50 inferences to minimize serial overhead)
                if (FAST_PLAYBACK && (inference_count % 50 == 0)) {
                  Serial.print("[Progress] Inferences: ");
                  Serial.print(inference_count);
                  Serial.print(" | Time: ");
                  Serial.print((float)processed_count / 100.0f, 0);  // No decimal to reduce print time
                  Serial.print("s | Last: ");
                  Serial.print(stage_str);
                  Serial.print(" (");
                  Serial.print(max_confidence * 100.0f, 0);  // No decimal
                  Serial.println("%)");
                }

                // Optional: Print to serial plot if enabled
                if (enable_serial_plot) {
                  Serial.print((float)processed_count / 100.0f, 3);
                  Serial.print(",CH0=");
                  Serial.print(eeg_sample[config.bipolar_channel_positive], 2);
                  Serial.print(",CH6=");
                  Serial.print(eeg_sample[config.bipolar_channel_negative], 2);
                  Serial.print(",Bipolar=");
                  Serial.print(bipolar_sample, 2);
                  Serial.print(",Processed100Hz=");
                  Serial.print(output_100hz, 2);
                  Serial.print(",");
                  Serial.print(stage_str);
                  Serial.print(",");
                  Serial.println(max_confidence, 3);
                }
              }
            }
        }
      }  // End if (sample_ready)
      
      // Check if we should stop (duration limit or end of file)
      if (sample_count >= (MAX_DURATION_SECONDS * config.sample_rate)) {
        Serial.println("Reached maximum test duration");
        while (1); // Stop here
      }
      
    } else {
      // End of file reached
      Serial.println("End of file reached");
      Serial.print("Total samples processed: ");
      Serial.println(sample_count);
      Serial.print("Total inferences: ");
      Serial.println(inference_count);
      while (1); // Stop here
    }
  }
}

void handleSerialCommands() {
  if (Serial.available()) {
    char cmd = Serial.read();

    switch (cmd) {
      case 'p':
        enable_serial_plot = !enable_serial_plot;
        Serial.print("Serial plotting: ");
        Serial.println(enable_serial_plot ? "ON" : "OFF");
        break;

      case 'i':
        enable_inference = !enable_inference;
        Serial.print("ML inference: ");
        Serial.println(enable_inference ? "ON" : "OFF");
        break;

      case 'd':
        debugLogger.setEnabled(!debugLogger.isEnabled());
        break;

      case 's':
        printStatistics();
        break;

      case 'r':
        Serial.println("Restarting playback...");
        eegReader.seekToTime(START_TIME_SECONDS);
        sample_count = 0;
        inference_count = 0;
        debug_100hz_index = 0;
        debug_buffer_full = false;
        last_sample_time = micros();
        break;

      default:
        Serial.println("Unknown command. Use p/i/d/s/r");
        break;
    }
  }
}

void printStatistics() {
  Serial.println("=== Statistics ===");
  Serial.print("Samples processed: ");
  Serial.println(sample_count);
  Serial.print("Current playback time: ");
  Serial.print((float)sample_count / config.sample_rate, 1);
  Serial.println(" seconds");
  Serial.print("Actual sample rate: ");
  Serial.print(sample_count / (millis() / 1000.0f), 1);
  Serial.println(" Hz");
  Serial.print("ML inferences: ");
  Serial.println(inference_count);
  Serial.print("File size: ");
  Serial.print(eegReader.getFileSize());
  Serial.println(" bytes");
  Serial.print("Estimated file duration: ");
  Serial.print(eegReader.getDurationSeconds(), 1);
  Serial.println(" seconds");
  Serial.println("==================");
}