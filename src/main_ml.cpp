#include <Arduino.h>
#include <SPI.h>
#include "ADS1299_Custom.h"
#include "MLInference.h"
#include "EEGProcessor.h"
#include "model.h"

// ADS1299 setup
ADS1299_Custom ads;

// ML components
MLInference mlInference;
EEGProcessor eegProcessor;

// Data buffers
float eeg_sample[ADS1299_CHANNELS];
float processed_window[MODEL_INPUT_SIZE];
float ml_output[MODEL_OUTPUT_SIZE];

// Timing
unsigned long last_inference_time = 0;
const unsigned long inference_interval = 4000; // Run inference every 4 seconds

// Statistics
unsigned long sample_count = 0;
unsigned long inference_count = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial) {}
  Serial.println("Sleep Headband TinyML Firmware Starting...");
  
  // Initialize SPI
  SPI.begin();
  
  // Initialize ADS1299
  Serial.println("Initializing ADS1299...");
  if (!ads.begin()) {
    Serial.println("Failed to initialize ADS1299");
    while (1);
  }
  
  // Start data acquisition
  ads.startAcquisition();
  Serial.println("ADS1299 initialized and acquisition started");
  
  // Initialize EEG processor
  if (!eegProcessor.begin()) {
    Serial.println("Failed to initialize EEG processor");
    while (1);
  }
  
  // Initialize ML inference
  Serial.println("Initializing TensorFlow Lite...");
  if (!mlInference.begin()) {
    Serial.println("Failed to initialize ML inference");
    while (1);
  }
  
  Serial.println("Setup complete. Starting data acquisition...");
  Serial.println("Format: Sample#, CH1, CH2, CH3, CH4, CH5, CH6, CH7, CH8, SleepStage, Confidence");
}

void loop() {
  // Check if new data is available from ISR
  int32_t rawData[ADS1299_CHANNELS];
  if (ads.getLatestData(rawData)) {
    // Convert raw ADC values to microvolts
    for (int i = 0; i < ADS1299_CHANNELS; i++) {
      // Use custom driver's conversion (already implements MATLAB formula)
      eeg_sample[i] = ads.convertToMicrovolts(rawData[i]);
    }
    
    // Add sample to processor
    eegProcessor.addSample(eeg_sample);
    sample_count++;
    
    // Print raw data (optional, comment out for performance)
    Serial.print(sample_count);
    for (int i = 0; i < ADS1299_CHANNELS; i++) {
      Serial.print(",");
      Serial.print(eeg_sample[i], 2);
    }
    
    // Check if we should run inference
    unsigned long current_time = millis();
    if (eegProcessor.isWindowReady() && 
        (current_time - last_inference_time) >= inference_interval) {
      
      // Get processed window for ML
      if (eegProcessor.getProcessedWindow(processed_window)) {
        // Run inference
        if (mlInference.predict(processed_window, ml_output)) {
          // Get predicted sleep stage
          SleepStage predicted_stage = mlInference.getPredictedStage(ml_output);
          
          // Find confidence (max probability)
          float max_confidence = 0.0f;
          for (int i = 0; i < MODEL_OUTPUT_SIZE; i++) {
            if (ml_output[i] > max_confidence) {
              max_confidence = ml_output[i];
            }
          }
          
          // Print sleep stage prediction
          Serial.print(",");
          switch (predicted_stage) {
            case WAKE: Serial.print("WAKE"); break;
            case LIGHT_SLEEP: Serial.print("LIGHT"); break;
            case DEEP_SLEEP: Serial.print("DEEP"); break;
            case REM_SLEEP: Serial.print("REM"); break;
            default: Serial.print("UNKNOWN"); break;
          }
          Serial.print(",");
          Serial.print(max_confidence, 3);
          
          inference_count++;
          last_inference_time = current_time;
          
          // Optional: Log to SD card
          // logToSD(predicted_stage, max_confidence, ml_output);
        } else {
          Serial.print(",ERROR,0.000");
        }
      } else {
        Serial.print(",PROCESSING,0.000");
      }
    } else {
      Serial.print(",,");
    }
    
    Serial.println();
  }
  
  // Print statistics every 30 seconds
  static unsigned long last_stats = 0;
  if (millis() - last_stats > 30000) {
    Serial.print("Stats - Samples: ");
    Serial.print(sample_count);
    Serial.print(", Inferences: ");
    Serial.print(inference_count);
    Serial.print(", Sample Rate: ");
    Serial.print(sample_count / (millis() / 1000.0f), 1);
    Serial.println(" Hz");
    last_stats = millis();
  }
}

// Optional: Function to log results to SD card
void logToSD(SleepStage stage, float confidence, float* probabilities) {
  // Implementation depends on your SD card setup
  // You could use the existing SD card code from the original project
}