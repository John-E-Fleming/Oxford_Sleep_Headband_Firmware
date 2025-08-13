#include <Arduino.h>
#include <SPI.h>
#include <SD.h>
#include "EEGFileReader.h"
#include "EEGProcessor.h"
#include "MLInference.h"
#include "model.h"

// SD card setup - try different approaches for Teensy 4.1
// For built-in SD, we might need to use different pins or SDIO

// EEG data components
EEGFileReader eegReader;
EEGProcessor eegProcessor;
MLInference mlInference;

// Data buffers
float eeg_sample[ADS1299_CHANNELS];
float processed_window[MODEL_INPUT_SIZE];
float ml_output[MODEL_OUTPUT_SIZE];

// Timing control for 4000Hz playback  
const unsigned long SAMPLE_INTERVAL_US = 250; // 1/4000Hz = 250µs
unsigned long last_sample_time = 0;

// Statistics and control
unsigned long sample_count = 0;
unsigned long inference_count = 0;
bool enable_inference = true;
bool enable_serial_plot = true;

// Test parameters
const float START_TIME_SECONDS = 0.0f;    // Start from beginning
const float MAX_DURATION_SECONDS = 300.0f; // Test for 5 minutes max

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
  Serial.println("  r - Restart playback");
  Serial.println();
  
  // Initialize SD card - try multiple approaches for Teensy 4.1
  Serial.println("Initializing SD card...");
  bool sd_ok = false;
  
  // Try approach 1: No pin specified (uses default)
  if (SD.begin()) {
    sd_ok = true;
    Serial.println("SD initialized with default settings");
  }
  // Try approach 2: SDIO interface (faster on Teensy 4.1)
  else if (SD.begin(BUILTIN_SDCARD)) {
    sd_ok = true;
    Serial.println("SD initialized with BUILTIN_SDCARD");
  }
  // Try approach 3: Standard SPI pins
  else if (SD.begin(10)) {
    sd_ok = true;
    Serial.println("SD initialized with pin 10");
  }
  
  if (!sd_ok) {
    Serial.println("SD card initialization failed - continuing without SD card for testing");
    Serial.println("Insert SD card with EEG file to enable file playback");
    Serial.println("Make sure SD card is properly inserted in Teensy 4.1 slot");
    enable_inference = false; // Disable since we can't read data
  } else {
    Serial.println("SD card initialized");
    
    // Open EEG file
    Serial.println("Opening EEG file...");
    if (!eegReader.begin("SdioLogger_miklos_night_2.bin")) {
      Serial.println("EEG file not found. Available files:");
      File root = SD.open("/");
      while (true) {
        File entry = root.openNextFile();
        if (!entry) break;
        Serial.print("  ");
        Serial.println(entry.name());
        entry.close();
      }
      root.close();
      enable_inference = false;
    } else {
      Serial.println("EEG file opened successfully");
    }
  }
  
  // Set format based on MATLAB analysis: int32, 9 channels
  eegReader.setFormat(FORMAT_INT32, 9);
  
  // Seek to start time if specified
  if (START_TIME_SECONDS > 0) {
    Serial.print("Seeking to ");
    Serial.print(START_TIME_SECONDS);
    Serial.println(" seconds...");
    eegReader.seekToTime(START_TIME_SECONDS);
  }
  
  // Initialize EEG processor
  if (!eegProcessor.begin()) {
    Serial.println("Failed to initialize EEG processor");
    while (1);
  }
  
  // Initialize ML inference (optional for initial testing)
  Serial.println("Initializing ML inference...");
  if (!mlInference.begin()) {
    Serial.println("ML inference initialization failed - continuing without ML");
    enable_inference = false;
  } else {
    Serial.println("ML inference initialized");
  }
  
  Serial.println("Starting EEG playback...");
  
  if (enable_serial_plot) {
    Serial.println("Serial plot format: Time,CH1,CH2,CH3,CH4,CH5,CH6,CH7,CH8,CH9[,Stage,Confidence]");
  }
  
  last_sample_time = micros();
}

void loop() {
  // Handle serial commands
  handleSerialCommands();
  
  // Check if it's time for the next sample (4000Hz timing)
  unsigned long current_time = micros();
  if (current_time - last_sample_time >= SAMPLE_INTERVAL_US) {
    
    // Read next sample from file - no synthetic data fallback
    bool has_data = false;
    
    if (SD.exists("SdioLogger_miklos_night_2.bin")) {
      has_data = eegReader.readNextSample(eeg_sample);
      if (!has_data) {
        Serial.println("ERROR: Failed to read from EEG file");
        Serial.println("Check file format and integrity");
        while(1); // Stop execution
      }
    } else {
      Serial.println("ERROR: EEG file 'SdioLogger_miklos_night_2.bin' not found on SD card");
      Serial.println("Available files:");
      // List files on SD card for debugging
      File root = SD.open("/");
      while (true) {
        File entry = root.openNextFile();
        if (!entry) break;
        Serial.print("  ");
        Serial.println(entry.name());
        entry.close();
      }
      root.close();
      while(1); // Stop execution
    }
    
    if (has_data) {
      sample_count++;
      last_sample_time = current_time;
      
      // Add sample to processor for ML inference
      eegProcessor.addSample(eeg_sample);
      
      // Serial plotting output
      if (enable_serial_plot) {
        // Time stamp (in seconds) - updated for 4000Hz
        Serial.print((float)sample_count / 4000.0f, 3);
        
        // EEG channel data
        for (int i = 0; i < ADS1299_CHANNELS; i++) {
          Serial.print(",");
          Serial.print(eeg_sample[i], 2);
        }
        
        // ML inference (if enabled and ready)
        if (enable_inference && eegProcessor.isWindowReady()) {
          static unsigned long last_inference = 0;
          if (millis() - last_inference > 4000) { // Every 4 seconds
            
            if (eegProcessor.getProcessedWindow(processed_window)) {
              if (mlInference.predict(processed_window, ml_output)) {
                SleepStage predicted_stage = mlInference.getPredictedStage(ml_output);
                
                float max_confidence = 0.0f;
                for (int i = 0; i < MODEL_OUTPUT_SIZE; i++) {
                  if (ml_output[i] > max_confidence) {
                    max_confidence = ml_output[i];
                  }
                }
                
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
                last_inference = millis();
              } else {
                Serial.print(",ERROR,0.000");
              }
            }
          }
        }
        
        Serial.println();
      }
      
      // Check if we should stop (duration limit or end of file)
      if (sample_count >= (MAX_DURATION_SECONDS * 4000)) {
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
        
      case 's':
        printStatistics();
        break;
        
      case 'r':
        Serial.println("Restarting playback...");
        eegReader.seekToTime(START_TIME_SECONDS);
        sample_count = 0;
        inference_count = 0;
        last_sample_time = micros();
        break;
        
      default:
        Serial.println("Unknown command. Use p/i/s/r");
        break;
    }
  }
}

void printStatistics() {
  Serial.println("=== Statistics ===");
  Serial.print("Samples processed: ");
  Serial.println(sample_count);
  Serial.print("Current playback time: ");
  Serial.print((float)sample_count / 4000.0f, 1);
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