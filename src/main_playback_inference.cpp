#include <Arduino.h>
#include <SPI.h>
#include <SdFat.h>
#include "EEGFileReader.h"
#include "EEGProcessor.h"
#include "MLInference.h"
#include "model.h"
#include "Config.h"
#include "EEGQualityChecker.h"
#include "InferenceLogger.h"
#include "DebugLogger.h"

// Preprocessing pipeline selection based on build flags
#if defined(USE_ALT_PREPROCESSING_A) || defined(USE_ALT_PREPROCESSING_B) || defined(USE_ALT_PREPROCESSING_C) || defined(USE_ALT_PREPROCESSING_D)
  #include "PreprocessingPipelineAlt.h"
  #define USING_ALT_PIPELINE 1
#else
  #include "PreprocessingPipeline.h"
  #define USING_ALT_PIPELINE 0
#endif

// Include validation support if enabled
#ifdef ENABLE_VALIDATION_MODE
#include "ValidationReader.h"
#endif

// SdFat object for EEG file reading (matching colleague's setup)
SdFat sd;

// Configuration
Config config;
unsigned long SAMPLE_INTERVAL_US = 250; // Will be set from config (4000Hz = 250us)

// EEG data components
EEGFileReader eegReader;
EEGProcessor eegProcessor;
MLInference mlInference;

// Preprocessing pipeline (selected at compile time)
#if USING_ALT_PIPELINE
PreprocessingPipelineAlt preprocessingPipeline;
#else
PreprocessingPipeline preprocessingPipeline;
#endif

EEGQualityChecker qualityChecker;
InferenceLogger inferenceLogger;
DebugLogger debugLogger;

// Validation support (optional, enabled with ENABLE_VALIDATION_MODE flag)
#ifdef ENABLE_VALIDATION_MODE
ValidationReader validationReader;
#endif

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
unsigned long last_inference_end_time = 0;  // For timing between inferences
unsigned long total_read_time_us = 0;       // Cumulative SD read time
unsigned long total_preprocess_time_us = 0; // Cumulative preprocessing time
unsigned long samples_since_timing_reset = 0;
bool enable_inference = true;
bool enable_serial_plot = !FAST_PLAYBACK;  // Disable plotting in fast mode (printing slows down processing)
bool enable_quality_check = true;
bool enable_inference_logging = true;   // ON by default to capture all epochs (toggle with 'l')
bool enable_timing_output = false;      // OFF by default (toggle with 't')

// EEG logging variables (for visualization)
SdFile eegLogFile;
bool eegLogFileOpen = false;
unsigned long eeg_sample_index = 0;

// Processing variables
unsigned long processed_count = 0;

// Test parameters
const float START_TIME_SECONDS = 0.0f;    // Start from beginning
const float MAX_DURATION_SECONDS = 30000.0f; // 8 hours 20 min buffer (ensures entire file is processed)

// Sliding window configuration
const int INFERENCE_WINDOW_SECONDS = 30;   // 30-second analysis window
const int INFERENCE_INTERVAL_SECONDS = 30; // Perform inference every 30 seconds (non-overlapping)
const float WINDOW_OVERLAP_PERCENT = 100.0f * (INFERENCE_WINDOW_SECONDS - INFERENCE_INTERVAL_SECONDS) / INFERENCE_WINDOW_SECONDS;

// Forward declarations
void handleSerialCommands();
void printStatistics();
void initializeLogging();
void closeLogging();

// Initialize logging files (called when 'l' command enables logging)
void initializeLogging() {
  if (!enable_inference_logging) return;

  // Extract base filename (remove path if present)
  String baseFilename = config.datafile;
  int lastSlash = baseFilename.lastIndexOf('/');
  if (lastSlash >= 0) {
    baseFilename = baseFilename.substring(lastSlash + 1);
  }
  baseFilename.replace(".bin", "");

  // Ensure /realtime_logs/ directory exists
  if (!sd.exists("/realtime_logs")) {
    sd.mkdir("/realtime_logs");
  }

  // Initialize prediction logger (writes to /realtime_logs/)
  String predFilename = baseFilename + "_predictions.csv";
  if (inferenceLogger.begin(predFilename)) {
    Serial.print("Prediction logging to: /realtime_logs/");
    Serial.println(predFilename);
  }

  // Initialize 100Hz EEG logger (writes to /realtime_logs/)
  String eegPath = "/realtime_logs/" + baseFilename + "_eeg_100hz.csv";
  if (eegLogFile.open(eegPath.c_str(), O_WRITE | O_CREAT | O_TRUNC)) {
    eegLogFile.println("sample_index,timestamp_ms,eeg_uv");
    eegLogFile.sync();
    eegLogFileOpen = true;
    eeg_sample_index = 0;
    Serial.print("EEG logging to: ");
    Serial.println(eegPath);
  } else {
    Serial.print("ERROR: Failed to open EEG log file: ");
    Serial.println(eegPath);
  }
}

// Close logging files
void closeLogging() {
  inferenceLogger.close();
  if (eegLogFileOpen) {
    eegLogFile.close();
    eegLogFileOpen = false;
    Serial.println("Log files closed");
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 5000) {} // Wait up to 5 seconds for serial
  
  Serial.println("=== EEG File Playback Test ===");
  Serial.println("Commands:");
  Serial.println("  p - Toggle serial plotting");
  Serial.println("  i - Toggle ML inference");
  Serial.println("  s - Show statistics");
  Serial.println("  l - Toggle logging (predictions + 100Hz EEG to SD) [ON by default]");
  Serial.println("  t - Toggle timing output");
  Serial.println("  d - Toggle debug logging (CSV output)");
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
      // Debug: confirm buffer status
      Serial.print("File reader buffer size: ");
      Serial.print(131072);  // BUFFER_SIZE from EEGFileReader.h
      Serial.println(" bytes (128KB)");
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

  // Configure preprocessing pipeline
#if USING_ALT_PIPELINE
  #if defined(USE_ALT_PREPROCESSING_A)
    preprocessingPipeline.setDownsampleMethod(PreprocessingPipelineAlt::DECIMATE);
    Serial.println("Using ALT preprocessing A: 4kHz->500Hz(decimate)->100Hz(avg)->filter@100Hz");
  #elif defined(USE_ALT_PREPROCESSING_B)
    preprocessingPipeline.setDownsampleMethod(PreprocessingPipelineAlt::AVERAGE);
    Serial.println("Using ALT preprocessing B: 4kHz->500Hz(average)->100Hz(avg)->filter@100Hz");
  #elif defined(USE_ALT_PREPROCESSING_C)
    preprocessingPipeline.setDownsampleMethod(PreprocessingPipelineAlt::DIRECT_DECIMATE);
    Serial.println("Using ALT preprocessing C: 4kHz->100Hz(decimate, every 40th)->filter@100Hz");
  #elif defined(USE_ALT_PREPROCESSING_D)
    preprocessingPipeline.setDownsampleMethod(PreprocessingPipelineAlt::DIRECT_AVERAGE);
    Serial.println("Using ALT preprocessing D: 4kHz->100Hz(average 40 samples)->filter@100Hz");
  #endif
#else
  Serial.println("Using STANDARD preprocessing: 4kHz->250Hz->filter@250Hz->100Hz");
#endif

  // Initialize logging (ON by default to capture all epochs)
  // Press 'l' to disable logging if not needed
  initializeLogging();

  // Initialize debug logger
  if (debugLogger.begin(&sd)) {
    Serial.println("Debug logger ready (use 'd' to enable)");
    // Set output filename based on input data file
    debugLogger.setOutputFilename(config.datafile.c_str());
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

  // Initialize validation mode (if enabled)
#ifdef ENABLE_VALIDATION_MODE
  Serial.println();
  Serial.println("=== VALIDATION MODE ENABLED ===");
  if (validationReader.begin("data/reference_predictions.csv", &sd)) {
    Serial.print("Validation ready with ");
    Serial.print(validationReader.getNumEpochs());
    Serial.println(" reference predictions");

    // Enable prediction logging to SD card for confusion matrix analysis
    if (validationReader.enablePredictionLogging("data/teensy_predictions.csv")) {
      Serial.println("Teensy predictions will be saved to: data/teensy_predictions.csv");
    }
  } else {
    Serial.println("WARNING: Validation mode enabled but failed to load reference predictions");
    Serial.println("Make sure data/reference_predictions.csv exists on SD card");
  }
  Serial.println("===============================");
  Serial.println();
#endif

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
    
    // File existence already verified at startup - just read directly
    unsigned long read_start = micros();
    has_data = eegReader.readNextSample(eeg_sample);
    total_read_time_us += (micros() - read_start);
    samples_since_timing_reset++;
    if (!has_data) {
      Serial.println("End of file reached or read error");
      Serial.print("Total samples processed: ");
      Serial.println(sample_count);

      // Print validation summary if enabled
#ifdef ENABLE_VALIDATION_MODE
      if (validationReader.isLoaded()) {
        validationReader.printSummary();
      }
#endif

      while(1); // Stop execution
    }
    
    if (has_data) {
      sample_count++;
      last_sample_time = current_time;

      // Create bipolar derivation (CH0 - CH6) for ML processing
      float bipolar_sample = eeg_sample[config.bipolar_channel_positive] - eeg_sample[config.bipolar_channel_negative];

      // DEBUG: Log first 20 raw bipolar samples at 4kHz (disabled for speed)
      // static bool logged_first_samples = false;
      // if (!logged_first_samples && sample_count <= 20) {
      //   Serial.print("[RAW 4kHz] Sample "); Serial.print(sample_count);
      //   Serial.print(": bipolar="); Serial.print(bipolar_sample, 2);
      //   Serial.print(" (CH"); Serial.print(config.bipolar_channel_positive);
      //   Serial.print("="); Serial.print(eeg_sample[config.bipolar_channel_positive], 2);
      //   Serial.print(" - CH"); Serial.print(config.bipolar_channel_negative);
      //   Serial.print("="); Serial.print(eeg_sample[config.bipolar_channel_negative], 2);
      //   Serial.println(")");
      //   if (sample_count == 20) logged_first_samples = true;
      // }

      // Process through complete pipeline: 4000Hz -> 250Hz -> BP filter -> 100Hz
      float output_100hz;
      unsigned long preprocess_start = micros();
      bool sample_ready = preprocessingPipeline.processSample(bipolar_sample, output_100hz);
      total_preprocess_time_us += (micros() - preprocess_start);

      // DEBUG: Log first 20 samples at 100Hz (disabled for speed)
      // static int logged_100hz_count = 0;
      // if (sample_ready && logged_100hz_count < 20) {
      //   Serial.print("[100Hz] Sample "); Serial.print(logged_100hz_count);
      //   Serial.print(": "); Serial.println(output_100hz, 4);
      //   logged_100hz_count++;
      // }

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

        // Log 100Hz sample (if logging enabled via 'l' command)
        if (enable_inference_logging && eegLogFileOpen) {
          eegLogFile.print(eeg_sample_index++);
          eegLogFile.print(",");
          eegLogFile.print((unsigned long)(processed_count * 10));  // ms at 100Hz
          eegLogFile.print(",");
          eegLogFile.println(output_100hz, 4);
          if (eeg_sample_index % 1000 == 0) eegLogFile.sync();  // Sync every 1000 samples (10s)
        }

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

              // Timing: measure data loading time (time since last inference ended)
              if (enable_timing_output && last_inference_end_time > 0) {
                unsigned long data_load_time_ms = (micros() - last_inference_end_time) / 1000;
                Serial.print("[TIMING] Data loading took ");
                Serial.print(data_load_time_ms);
                Serial.print(" ms (");
                Serial.print(samples_since_timing_reset);
                Serial.println(" samples)");

                // Breakdown of time spent
                Serial.print("[TIMING]   SD read: ");
                Serial.print(total_read_time_us / 1000);
                Serial.print(" ms, Preprocess: ");
                Serial.print(total_preprocess_time_us / 1000);
                Serial.print(" ms, Other: ");
                Serial.print(data_load_time_ms - (total_read_time_us + total_preprocess_time_us) / 1000);
                Serial.println(" ms");
              }

              // Reset timing counters (always, even if not printing)
              total_read_time_us = 0;
              total_preprocess_time_us = 0;
              samples_since_timing_reset = 0;

              unsigned long inference_start = micros();
              if (mlInference.predict(processed_window, ml_output, inference_count)) {
                unsigned long inference_time_ms = (micros() - inference_start) / 1000;
                if (enable_timing_output) {
                  Serial.print("[TIMING] Inference took ");
                  Serial.print(inference_time_ms);
                  Serial.println(" ms");
                }
                last_inference_end_time = micros();  // Mark when inference finished

                SleepStage predicted_stage = mlInference.getPredictedStage(ml_output);

                // === CHECKPOINT DEBUG: Output statistics for first N epochs ===
                // Format designed for tools/compare_teensy_python.py
                // Set CHECKPOINT_DEBUG_EPOCHS to 0 to disable
                #define CHECKPOINT_DEBUG_EPOCHS 5  // Enable for first 5 epochs
                if (inference_count < CHECKPOINT_DEBUG_EPOCHS) {

                  // CHECKPOINT A: 100Hz preprocessed signal (before normalization)
                  if (debug_buffer_full) {
                    float pre_sum = 0, pre_sum_sq = 0;
                    float pre_min = 1e10f, pre_max = -1e10f;
                    float first_10[10], last_10[10];

                    for (int i = 0; i < 3000; i++) {
                      int buf_idx = (debug_100hz_index + i) % DEBUG_BUFFER_SIZE;
                      float val = debug_100hz_buffer[buf_idx];
                      pre_sum += val;
                      pre_sum_sq += val * val;
                      if (val < pre_min) pre_min = val;
                      if (val > pre_max) pre_max = val;
                      if (i < 10) first_10[i] = val;
                      if (i >= 2990) last_10[i - 2990] = val;
                    }
                    float pre_mean = pre_sum / 3000.0f;
                    float pre_std = sqrt((pre_sum_sq / 3000.0f) - (pre_mean * pre_mean));

                    Serial.print("[CHECKPOINT A] 100Hz preprocessed - Epoch ");
                    Serial.println(inference_count);
                    Serial.print("mean="); Serial.print(pre_mean, 4);
                    Serial.print(" std="); Serial.print(pre_std, 4);
                    Serial.print(" min="); Serial.print(pre_min, 4);
                    Serial.print(" max="); Serial.println(pre_max, 4);
                    Serial.print("first_10:");
                    for (int i = 0; i < 10; i++) { Serial.print(" "); Serial.print(first_10[i], 2); }
                    Serial.println();
                    Serial.print("last_10:");
                    for (int i = 0; i < 10; i++) { Serial.print(" "); Serial.print(last_10[i], 2); }
                    Serial.println();
                  }

                  // CHECKPOINT B: Epoch extraction boundaries
                  int start_sample = inference_count * 3000;
                  int end_sample = start_sample + 3000;
                  Serial.print("[CHECKPOINT B] Epoch extraction - Epoch ");
                  Serial.println(inference_count);
                  Serial.print("start_sample="); Serial.print(start_sample);
                  Serial.print(" end_sample="); Serial.println(end_sample);

                  // CHECKPOINT C: Normalization statistics
                  float norm_sum = 0, norm_sum_sq = 0;
                  for (int i = 0; i < 3000; i++) {
                    norm_sum += processed_window[i];
                    norm_sum_sq += processed_window[i] * processed_window[i];
                  }
                  float norm_mean = norm_sum / 3000.0f;
                  float norm_std = sqrt((norm_sum_sq / 3000.0f) - (norm_mean * norm_mean));

                  Serial.print("[CHECKPOINT C] Normalization - Epoch ");
                  Serial.println(inference_count);
                  Serial.print("mean="); Serial.print(norm_mean, 8);
                  Serial.print(" std="); Serial.println(norm_std, 8);
                  Serial.print("first_10:");
                  for (int i = 0; i < 10; i++) { Serial.print(" "); Serial.print(processed_window[i], 6); }
                  Serial.println();
                  Serial.print("last_10:");
                  for (int i = 2990; i < 3000; i++) { Serial.print(" "); Serial.print(processed_window[i], 6); }
                  Serial.println();

                  // CHECKPOINT D: Model input info
                  // Note: This model uses FLOAT32 inputs (not INT8 quantized)
                  float scale;
                  int32_t zero_point;
                  bool has_quant = mlInference.getInputQuantizationParams(scale, zero_point);
                  Serial.print("[CHECKPOINT D] Model input - Epoch ");
                  Serial.println(inference_count);
                  if (has_quant && scale > 0.0f) {
                    // INT8 quantized model
                    Serial.println("input_type=INT8");
                    Serial.print("scale="); Serial.print(scale, 10);
                    Serial.print(" zero_point="); Serial.println(zero_point);
                  } else {
                    // FLOAT32 model - no quantization needed
                    Serial.println("input_type=FLOAT32 (no quantization)");
                  }
                  Serial.print("first_10_float:");
                  for (int i = 0; i < 10; i++) {
                    Serial.print(" "); Serial.print(processed_window[i], 4);
                  }
                  Serial.println();
                  Serial.print("epoch_index="); Serial.print(inference_count);
                  Serial.print(" scaled="); Serial.println(static_cast<float>(inference_count) / 1000.0f, 6);

                  // Model output
                  Serial.println("[MODEL OUTPUT]");
                  Serial.print("probs:");
                  for (int i = 0; i < MODEL_OUTPUT_SIZE; i++) {
                    Serial.print(" "); Serial.print(ml_output[i], 4);
                  }
                  Serial.println();
                  Serial.println();
                }
                // === END CHECKPOINT DEBUG ===

                float max_confidence = 0.0f;
                for (int i = 0; i < MODEL_OUTPUT_SIZE; i++) {
                  if (ml_output[i] > max_confidence) {
                    max_confidence = ml_output[i];
                  }
                }

                // Convert predicted stage to string for debug logging
                const char* stage_str = "UNKNOWN";
                switch (predicted_stage) {
                  case WAKE: stage_str = "WAKE"; break;            // yy0
                  case N1_VERY_LIGHT: stage_str = "N1"; break;    // yy1
                  case N2_LIGHT_SLEEP: stage_str = "N2"; break;   // yy2
                  case N3_DEEP_SLEEP: stage_str = "N3"; break;    // yy3
                  case REM_SLEEP: stage_str = "REM"; break;        // yy4
                }

                // Debug logging: Log model output probabilities
                if (debugLogger.isEnabled()) {
                  float time_s = processed_count / 100.0f;  // Time in seconds at 100Hz
                  debugLogger.logModelOutput(ml_output, MODEL_OUTPUT_SIZE, inference_count, time_s, stage_str);
                }

                // Log prediction to CSV (if logging enabled via 'l' command)
                if (enable_inference_logging && inferenceLogger.isLogging()) {
                  float epoch_end_seconds = (inference_count + 1) * INFERENCE_INTERVAL_SECONDS;
                  inferenceLogger.logPrediction(inference_count, epoch_end_seconds, ml_output);
                }

                // Validation: Compare against reference predictions (if enabled)
#ifdef ENABLE_VALIDATION_MODE
                if (validationReader.isLoaded()) {
                  validationReader.compareAndLog(inference_count, ml_output, stage_str);
                }
#endif

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

        // Close log files if open
        closeLogging();

        // Print validation summary if enabled
#ifdef ENABLE_VALIDATION_MODE
        if (validationReader.isLoaded()) {
          validationReader.printSummary();
        }
#endif

        while (1); // Stop here
      }
      
    } else {
      // End of file reached
      Serial.println("End of file reached");
      Serial.print("Total samples processed: ");
      Serial.println(sample_count);
      Serial.print("Total inferences: ");
      Serial.println(inference_count);

      // Print validation summary if enabled
#ifdef ENABLE_VALIDATION_MODE
      if (validationReader.isLoaded()) {
        validationReader.printSummary();
      }
#endif

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
        eeg_sample_index = 0;
        last_sample_time = micros();
        break;

      case 'l':
        enable_inference_logging = !enable_inference_logging;
        Serial.print("Logging (predictions + EEG): ");
        Serial.println(enable_inference_logging ? "ON" : "OFF");
        if (enable_inference_logging) {
          initializeLogging();
        } else {
          closeLogging();
        }
        break;

      case 't':
        enable_timing_output = !enable_timing_output;
        Serial.print("Timing output: ");
        Serial.println(enable_timing_output ? "ON" : "OFF");
        break;

      default:
        Serial.println("Unknown command. Use p/i/d/s/r/l/t");
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