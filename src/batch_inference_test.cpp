#include <Arduino.h>
#include <SPI.h>
#include <SdFat.h>
#include "EEGFileReader.h"
#include "EEGProcessor.h"
#include "MLInference.h"
#include "model.h"
#include "Config.h"
#include "BandpassFilter.h"
#include "EEGQualityChecker.h"
#include "InferenceLogger.h"
#include "DataLogger.h"

// Fast batch processing mode for model testing
// Processes entire EEG file as quickly as possible to verify model performance

// SdFat object for EEG file reading
SdFat sd;

// Configuration
Config config;

// EEG data components
EEGFileReader eegReader;
EEGProcessor eegProcessor;
MLInference mlInference;
BandpassFilter bandpassFilter;
EEGQualityChecker qualityChecker;
InferenceLogger inferenceLogger;
DataLogger dataLogger;

// Data buffers - allocate in external memory to save RAM1
float eeg_sample[ADS1299_CHANNELS];
float* processed_window = nullptr;  // Will be allocated in setup()
float ml_output[MODEL_OUTPUT_SIZE];

// Batch processing parameters
const bool ENABLE_QUALITY_CHECK = false;  // Disabled to see raw model outputs
const bool ENABLE_LOGGING = true;   // Re-enabled with streaming writes
const bool ENABLE_DATA_LOGGING = true;  // Log raw and normalized data for comparison
const bool USE_DUMMY_MODEL = false;   // Using real TensorFlow Lite model
const float BATCH_WINDOW_SECONDS = 30;    // 30-second windows (model expects this)
const float BATCH_SLIDE_SECONDS = 30;  // Non-overlapping windows (0% overlap)
//const float MAX_PROCESSING_TIME_SECONDS = 60.0f;   // Process up to 1 minute (reduced for memory)
const float MAX_PROCESSING_TIME_SECONDS = 36000.0f;   // Process up to 10 hours

// Statistics
unsigned long total_samples_processed = 0;
unsigned long total_inferences_attempted = 0;
unsigned long successful_inferences = 0;
unsigned long quality_rejections = 0;
unsigned long total_processing_time_ms = 0;

// Function declarations
void runBatchProcessing();
void printFinalResults();
String getStageName(SleepStage stage);
float getMaxConfidence(float* output);

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 5000) {} // Wait up to 5 seconds for serial
  
  Serial.println("=== Batch EEG Inference Test ===");
  Serial.println("Fast processing mode for model validation");
  Serial.println();
  
  // Allocate large buffers in external memory
  processed_window = (float*)extmem_malloc(MODEL_INPUT_SIZE * sizeof(float));
  if (!processed_window) {
    Serial.println("Failed to allocate processed_window in external memory");
    while (1);
  }
  Serial.println("Allocated processed_window in external memory");
  
  // Initialize SD card
  Serial.println("Initializing SD card...");
  bool sd_ok = false;
  
  if (sd.begin(SdioConfig(FIFO_SDIO))) {
    sd_ok = true;
    Serial.println("SD initialized with FIFO_SDIO");
  } else if (sd.begin(SdioConfig(DMA_SDIO))) {
    sd_ok = true;
    Serial.println("SD initialized with DMA_SDIO");
  } else if (sd.begin(SdSpiConfig(10, DEDICATED_SPI, SD_SCK_MHZ(50)))) {
    sd_ok = true;
    Serial.println("SD initialized with SPI");
  }
  
  if (!sd_ok) {
    Serial.println("SD card initialization failed");
    while (1);
  }
  
  // Load configuration
  Serial.println("Loading configuration...");
  SdFile rootDir;
  if (rootDir.open("/") && loadConfig(rootDir, config)) {
    Serial.println("Configuration loaded successfully");
    rootDir.close();
  } else {
    Serial.println("Failed to load config.txt, using defaults");
    config.datafile = "SdioLogger_miklos_night_2_Fs_100Hz.bin";
    config.sample_rate = 100;
    config.channels = 9;
    config.bipolar_channel_positive = 0;
    config.bipolar_channel_negative = 6;
  }
  
  Serial.print("Data file: ");
  Serial.println(config.datafile);
  Serial.print("Sample rate: ");
  Serial.print(config.sample_rate);
  Serial.println(" Hz");
  
  // Open EEG file
  Serial.println("Opening EEG file...");
  if (!eegReader.begin(config.datafile)) {
    Serial.println("Failed to open EEG file");
    while (1);
  }
  
  // Set format based on config
  EEGDataFormat format = FORMAT_INT32;
  if (config.format == "float32") format = FORMAT_FLOAT32;
  else if (config.format == "int16") format = FORMAT_INT16;
  eegReader.setFormat(format, config.channels);
  
  // Initialize EEG processor with batch parameters
  if (!eegProcessor.begin()) {
    Serial.println("Failed to initialize EEG processor");
    while (1);
  }
  eegProcessor.configureSlidingWindow(BATCH_WINDOW_SECONDS, BATCH_SLIDE_SECONDS);
  
  // Initialize ML inference
  Serial.print("Initializing ML inference (");
  Serial.print(USE_DUMMY_MODEL ? "dummy mode" : "real model");
  Serial.println(")...");
  if (!mlInference.begin(USE_DUMMY_MODEL)) {
    Serial.println("Failed to initialize ML inference");
    while (1);
  }
  
  // Initialize quality checker and logger
  if (ENABLE_LOGGING) {
    String log_filename = "batch_test_" + String(millis()) + ".csv";
    if (!inferenceLogger.begin(log_filename)) {
      Serial.println("Warning: Failed to initialize streaming logger");
    }
  }

  // Initialize data logger for raw and normalized data
  if (ENABLE_DATA_LOGGING) {
    String session_name = "batch_" + String(millis());
    if (!dataLogger.begin(session_name)) {
      Serial.println("Warning: Failed to initialize data logger");
    } else {
      Serial.println("Data logging enabled - will save raw and normalized data");
    }
  }

  Serial.println("Initialization complete. Starting batch processing...");
  Serial.println("Processing will be as fast as possible (no real-time constraints)");
  Serial.println();
  
  unsigned long start_time = millis();
  runBatchProcessing();
  total_processing_time_ms = millis() - start_time;
  
  // Print final results
  printFinalResults();
}

void runBatchProcessing() {
  unsigned long sample_count = 0;
  unsigned long window_start_sample = 0;
  
  Serial.println("Starting batch processing...");
  
  while (true) {
    // Read next sample
    bool has_data = eegReader.readNextSample(eeg_sample);
    
    if (!has_data) {
      Serial.println("End of file reached");
      break;
    }
    
    sample_count++;
    total_samples_processed++;
    
    // Create bipolar derivation
    float bipolar_sample = eeg_sample[config.bipolar_channel_positive] - eeg_sample[config.bipolar_channel_negative];
    
    // Apply bandpass filter
    float filtered_sample = bandpassFilter.process(bipolar_sample);

    // Log raw filtered sample (after bandpass, before normalization)
    if (ENABLE_DATA_LOGGING) {
      dataLogger.logRawSample(filtered_sample);
    }

    // Add to EEG processor
    eegProcessor.addFilteredSample(filtered_sample);
    
    // Check if we can perform inference
    if (eegProcessor.isInferenceTimeReady()) {
      total_inferences_attempted++;
      
      if (eegProcessor.getProcessedWindow(processed_window)) {
        bool quality_pass = true;
        EEGQualityChecker::QualityMetrics quality;

        // Log normalized window before inference
        if (ENABLE_DATA_LOGGING) {
          dataLogger.logNormalizedWindow(processed_window, MODEL_EEG_SAMPLES, successful_inferences);
        }

        // Optional quality check
        if (ENABLE_QUALITY_CHECK) {
          quality = qualityChecker.checkWindowQuality(
              processed_window, MODEL_INPUT_SIZE,
              eegProcessor.getFilteredMean(),
              eegProcessor.getFilteredStd());
          quality_pass = quality.is_valid;
        }

        if (quality_pass) {
          // Run inference on full 30-second window (3000 samples at 100Hz)
          // The predict function expects z-score normalized data + epoch index

          // Run inference with epoch index
          unsigned long inference_start = micros();
          bool inference_success = mlInference.predict(processed_window, ml_output, successful_inferences);
          unsigned long inference_time = micros() - inference_start;
          
          if (inference_success) {
            SleepStage predicted_stage = mlInference.getPredictedStage(ml_output);
            successful_inferences++;
            
            // Log if enabled
            if (ENABLE_LOGGING) {
              inferenceLogger.logInference(
                  millis(), window_start_sample, sample_count, successful_inferences,
                  ml_output, predicted_stage,
                  eegProcessor.getFilteredMean(), eegProcessor.getFilteredStd());
            }
            
            // Print progress every 10 inferences
            if (successful_inferences % 10 == 0) {
              Serial.print("Inference #");
              Serial.print(successful_inferences);
              Serial.print(": ");
              Serial.print(getStageName(predicted_stage));
              Serial.print(" (conf: ");
              Serial.print(getMaxConfidence(ml_output), 3);
              Serial.print(", time: ");
              Serial.print(inference_time);
              Serial.println("μs)");
            }
          }
        } else {
          // Quality rejection
          quality_rejections++;
          
          if (ENABLE_LOGGING) {
            inferenceLogger.logQualityRejection(
                millis(), window_start_sample, sample_count, total_inferences_attempted,
                quality, eegProcessor.getFilteredMean(), eegProcessor.getFilteredStd());
          }
          
          if (quality_rejections % 5 == 0) {
            Serial.print("Quality rejection #");
            Serial.print(quality_rejections);
            Serial.print(": ");
            Serial.println(quality.rejection_reason);
          }
        }
        
        // Reset inference counter
        eegProcessor.markInferenceComplete();
        window_start_sample = sample_count;
      }
    }
    
    // Check if we should stop (time limit)
    if (sample_count >= (MAX_PROCESSING_TIME_SECONDS * config.sample_rate)) {
      Serial.println("Reached processing time limit");
      break;
    }
    
    // Progress indicator
    if (sample_count % (config.sample_rate * 10) == 0) {  // Every 10 seconds
      Serial.print("Processed ");
      Serial.print(sample_count / config.sample_rate);
      Serial.print("s of data, ");
      Serial.print(successful_inferences);
      Serial.println(" inferences");
    }
  }
}

void printFinalResults() {
  Serial.println("\\n=== Batch Processing Complete ===");
  Serial.print("Total samples processed: ");
  Serial.println(total_samples_processed);
  Serial.print("Processing time: ");
  Serial.print(total_processing_time_ms / 1000.0f, 2);
  Serial.println(" seconds");
  Serial.print("Processing speed: ");
  Serial.print(total_samples_processed / (total_processing_time_ms / 1000.0f), 1);
  Serial.println(" samples/sec");
  
  Serial.print("Total inference attempts: ");
  Serial.println(total_inferences_attempted);
  Serial.print("Successful inferences: ");
  Serial.println(successful_inferences);
  Serial.print("Quality rejections: ");
  Serial.println(quality_rejections);
  
  if (total_inferences_attempted > 0) {
    Serial.print("Success rate: ");
    Serial.print(100.0f * successful_inferences / total_inferences_attempted, 1);
    Serial.println("%");
  }
  
  if (successful_inferences > 0) {
    Serial.print("Average inference time: ");
    Serial.print(mlInference.getLastInferenceTime());
    Serial.println(" μs");
  }
  
  // Print detailed statistics if logging enabled
  if (ENABLE_LOGGING) {
    Serial.println();
    inferenceLogger.printSummary();

    // Close the streaming log file
    inferenceLogger.close();
    Serial.println("Inference log file has been saved to SD card");
  }

  // Close data logger
  if (ENABLE_DATA_LOGGING) {
    Serial.println();
    dataLogger.close();
    Serial.println("Raw and normalized data files have been saved to SD card");
  }

  Serial.println("=====================================");
}

String getStageName(SleepStage stage) {
  switch (stage) {
    case N3_DEEP_SLEEP: return "N3";      // yy0 - Deep sleep (N3)
    case N2_LIGHT_SLEEP: return "N2";     // yy1 - Light sleep (N2)
    case N1_VERY_LIGHT: return "N1";      // yy2 - Very light sleep (N1)
    case REM_SLEEP: return "REM";         // yy3 - REM sleep
    case WAKE: return "WAKE";             // yy4 - Wake
    default: return "UNKNOWN";
  }
}

float getMaxConfidence(float* output) {
  float max_val = output[0];
  for (int i = 1; i < MODEL_OUTPUT_SIZE; i++) {
    if (output[i] > max_val) {
      max_val = output[i];
    }
  }
  return max_val;
}

void loop() {
  // All processing happens in setup() for batch mode
  // This keeps the system idle after completion
  delay(1000);
}