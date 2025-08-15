#include <Arduino.h>
#include <SPI.h>
#include <SdFat.h>
#include "EEGFileReader.h"
#include "EEGProcessor.h"
#include "MLInference.h"
#include "model.h"
#include "Config.h"
#include "BandpassFilter.h"

// SdFat object for EEG file reading (matching colleague's setup)
SdFat sd;

// Configuration
Config config;
unsigned long SAMPLE_INTERVAL_US = 4000; // Will be set from config

// EEG data components
EEGFileReader eegReader;
EEGProcessor eegProcessor;
MLInference mlInference;
BandpassFilter bandpassFilter;

// Data buffers
float eeg_sample[ADS1299_CHANNELS];
float processed_window[MODEL_INPUT_SIZE];
float ml_output[MODEL_OUTPUT_SIZE];

unsigned long last_sample_time = 0;

// Statistics and control
unsigned long sample_count = 0;
unsigned long inference_count = 0;
bool enable_inference = true;
bool enable_serial_plot = true;

// Processing variables
unsigned long processed_count = 0;

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
      Serial.println(" Hz");
      Serial.print("  Bipolar channels: ");
      Serial.print(config.bipolar_channel_positive);
      Serial.print(" - ");
      Serial.println(config.bipolar_channel_negative);
      
      // Set sample interval based on config
      SAMPLE_INTERVAL_US = 1000000 / config.sample_rate; // Convert Hz to microseconds
      
      rootDir.close();
    } else {
      Serial.println("Failed to load config.txt, using defaults");
      config.datafile = "SdioLogger_miklos_night_2_Fs_250Hz.bin";
      config.sample_rate = 100;
      config.channels = 9;
      config.bipolar_channel_positive = 0;
      config.bipolar_channel_negative = 6;
      SAMPLE_INTERVAL_US = 10000; // 100Hz
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
      processed_count++;
      
      // Apply 0.5-40Hz bandpass filter
      float filtered_sample = bandpassFilter.process(bipolar_sample);
      
      // Add to ML processor buffer (single channel)
      // For now, still using the old multi-channel approach
      eegProcessor.addSample(eeg_sample);
      
      // Serial plotting output
      if (enable_serial_plot) {
        // Time stamp (in seconds) - using config sample rate
        Serial.print((float)sample_count / config.sample_rate, 3);
        
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