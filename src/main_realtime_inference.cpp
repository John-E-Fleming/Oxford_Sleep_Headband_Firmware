/**
 * @file main_realtime_inference.cpp
 * @brief Real-time EEG sleep stage classification with live ADS1299 acquisition
 *
 * This file implements real-time sleep stage classification using the validated
 * preprocessing pipeline. It reads EEG data from ADS1299 hardware, processes it
 * through the same pipeline validated in playback mode, and outputs predictions.
 *
 * Data Flow:
 *   ADS1299 (4kHz, 8ch) → convertToMicrovolts → bipolar (CH0-CH6) →
 *   PreprocessingPipeline (4kHz→100Hz) → EEGProcessor (30s window) →
 *   MLInference → Sleep Stage
 *
 * Use this mode for: Production use with actual EEG headset
 *
 * To enable this mode, uncomment the REAL-TIME MODE line in platformio.ini
 */

#include <Arduino.h>
#include <SPI.h>
#include "ADS1299_Custom.h"
#include "MLInference.h"
#include "EEGProcessor.h"
#include "PreprocessingPipeline.h"
#include "model.h"

// ============================================================================
// Configuration - Adjust these settings to match your hardware setup
// ============================================================================

// Bipolar derivation channels (matches validated playback mode)
// Default: CH0 (F3) - CH6 (M2) for standard sleep EEG montage
const int BIPOLAR_CHANNEL_POSITIVE = 0;  // Positive electrode (e.g., F3)
const int BIPOLAR_CHANNEL_NEGATIVE = 6;  // Reference electrode (e.g., M2)

// Inference window configuration (must match model training)
const int INFERENCE_WINDOW_SECONDS = 30;   // 30-second analysis window
const int INFERENCE_INTERVAL_SECONDS = 30; // Non-overlapping windows

// Serial output control
const bool ENABLE_SAMPLE_PRINTING = false;  // Set true to print every sample (slow)
const bool ENABLE_VERBOSE_INFERENCE = true; // Print detailed inference results

// ============================================================================
// Global Objects
// ============================================================================

// Hardware interface
ADS1299_Custom ads;

// ML components (same as validated playback mode)
MLInference mlInference;
EEGProcessor eegProcessor;
PreprocessingPipeline preprocessingPipeline;

// Data buffers
float eeg_sample[ADS1299_CHANNELS];
float processed_window[MODEL_INPUT_SIZE];
float ml_output[MODEL_OUTPUT_SIZE];

// Statistics
unsigned long sample_count = 0;
unsigned long inference_count = 0;
unsigned long processed_100hz_count = 0;

// ============================================================================
// Setup
// ============================================================================

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 5000) {}  // Wait up to 5 seconds for serial

  Serial.println("===========================================");
  Serial.println("Sleep Headband - Real-Time Inference Mode");
  Serial.println("===========================================");
  Serial.println();

  // Initialize SPI for ADS1299 communication
  SPI.begin();

  // Initialize ADS1299 EEG acquisition
  Serial.println("Initializing ADS1299 EEG acquisition...");
  if (!ads.begin()) {
    Serial.println("ERROR: Failed to initialize ADS1299");
    Serial.println("Check hardware connections:");
    Serial.println("  - CS pin: " + String(ADS1299_CS1));
    Serial.println("  - DRDY pin: " + String(ADS1299_DRDY));
    Serial.println("  - START pin: " + String(ADS1299_START_PIN));
    Serial.println("  - PWDN pin: " + String(ADS1299_PWDN));
    while (1) { delay(1000); }
  }
  Serial.println("ADS1299 initialized successfully");

  // Start data acquisition at 4000Hz (matches preprocessing pipeline input)
  ads.startAcquisition();
  Serial.println("Data acquisition started at 4000Hz");

  // Initialize EEG processor with 30-second windows (matches validated mode)
  if (!eegProcessor.begin()) {
    Serial.println("ERROR: Failed to initialize EEG processor");
    while (1) { delay(1000); }
  }
  eegProcessor.configureSlidingWindow(INFERENCE_WINDOW_SECONDS, INFERENCE_INTERVAL_SECONDS);
  Serial.print("EEG processor configured: ");
  Serial.print(INFERENCE_WINDOW_SECONDS);
  Serial.print("s window, ");
  Serial.print(INFERENCE_INTERVAL_SECONDS);
  Serial.println("s interval");

  // Initialize ML inference engine
  Serial.println("Initializing TensorFlow Lite Micro...");
  if (!mlInference.begin()) {
    Serial.println("ERROR: Failed to initialize ML inference");
    while (1) { delay(1000); }
  }
  Serial.print("ML inference ready (");
  Serial.print(mlInference.isUsingRealModel() ? "real model" : "dummy model");
  Serial.println(")");

  // Print configuration summary
  Serial.println();
  Serial.println("Configuration:");
  Serial.print("  Bipolar derivation: CH");
  Serial.print(BIPOLAR_CHANNEL_POSITIVE);
  Serial.print(" - CH");
  Serial.println(BIPOLAR_CHANNEL_NEGATIVE);
  Serial.println("  Pipeline: 4kHz → 250Hz → BP filter → 100Hz → 30s window → CNN");
  Serial.println("  Model input: 3000 samples (30s @ 100Hz) + epoch index");
  Serial.println("  Model output: 5 classes (Wake, N1, N2, N3, REM)");
  Serial.println();
  Serial.println("===========================================");
  Serial.println("Starting real-time sleep classification...");
  Serial.println("===========================================");
  Serial.println();
}

// ============================================================================
// Main Loop
// ============================================================================

void loop() {
  // Check if new data is available from ADS1299 ISR
  int32_t rawData[ADS1299_CHANNELS];

  if (ads.getLatestData(rawData)) {
    sample_count++;

    // Convert raw ADC values to microvolts
    for (int i = 0; i < ADS1299_CHANNELS; i++) {
      eeg_sample[i] = ads.convertToMicrovolts(rawData[i]);
    }

    // Create bipolar derivation (matches validated playback mode)
    // This computes: output = CH_positive - CH_negative
    float bipolar_sample = eeg_sample[BIPOLAR_CHANNEL_POSITIVE] - eeg_sample[BIPOLAR_CHANNEL_NEGATIVE];

    // Process through complete validated pipeline: 4000Hz → 250Hz → BP filter → 100Hz
    float output_100hz;
    bool sample_ready = preprocessingPipeline.processSample(bipolar_sample, output_100hz);

    if (sample_ready) {
      // New 100Hz sample ready - add to ML processor buffer
      processed_100hz_count++;
      eegProcessor.addFilteredSample(output_100hz);

      // Check if we have a full 30-second window and it's time for inference
      if (eegProcessor.isInferenceTimeReady()) {

        // Get the processed window (Z-score normalized)
        if (eegProcessor.getProcessedWindow(processed_window)) {

          // Run ML inference with epoch index (required by model)
          unsigned long inference_start = micros();
          if (mlInference.predict(processed_window, ml_output, inference_count)) {
            unsigned long inference_time_us = micros() - inference_start;

            // Get predicted sleep stage
            SleepStage predicted_stage = mlInference.getPredictedStage(ml_output);

            // Find confidence (max probability)
            float max_confidence = 0.0f;
            for (int i = 0; i < MODEL_OUTPUT_SIZE; i++) {
              if (ml_output[i] > max_confidence) {
                max_confidence = ml_output[i];
              }
            }

            // Convert stage to string
            const char* stage_str = "UNKNOWN";
            switch (predicted_stage) {
              case WAKE: stage_str = "WAKE"; break;
              case N1_VERY_LIGHT: stage_str = "N1"; break;
              case N2_LIGHT_SLEEP: stage_str = "N2"; break;
              case N3_DEEP_SLEEP: stage_str = "N3"; break;
              case REM_SLEEP: stage_str = "REM"; break;
            }

            // Print inference result
            if (ENABLE_VERBOSE_INFERENCE) {
              Serial.println();
              Serial.print("[Epoch ");
              Serial.print(inference_count);
              Serial.print("] Stage: ");
              Serial.print(stage_str);
              Serial.print(" (");
              Serial.print(max_confidence * 100.0f, 1);
              Serial.print("%) | Inference: ");
              Serial.print(inference_time_us / 1000);
              Serial.print("ms | Time: ");
              Serial.print((float)sample_count / 4000.0f, 1);
              Serial.println("s");

              // Print probability distribution
              Serial.print("  Probs: W=");
              Serial.print(ml_output[0], 3);
              Serial.print(" N1=");
              Serial.print(ml_output[1], 3);
              Serial.print(" N2=");
              Serial.print(ml_output[2], 3);
              Serial.print(" N3=");
              Serial.print(ml_output[3], 3);
              Serial.print(" REM=");
              Serial.println(ml_output[4], 3);
            } else {
              // Compact output
              Serial.print(inference_count);
              Serial.print(",");
              Serial.print(stage_str);
              Serial.print(",");
              Serial.println(max_confidence, 3);
            }

            inference_count++;
            eegProcessor.markInferenceComplete();
          } else {
            Serial.println("[ERROR] ML inference failed");
          }
        }
      }
    }

    // Optional: Print every sample (very slow, for debugging only)
    if (ENABLE_SAMPLE_PRINTING) {
      Serial.print(sample_count);
      Serial.print(",");
      Serial.print(bipolar_sample, 2);
      Serial.println();
    }
  }

  // Print statistics every 60 seconds
  static unsigned long last_stats = 0;
  if (millis() - last_stats > 60000) {
    Serial.println();
    Serial.println("--- Statistics ---");
    Serial.print("Uptime: ");
    Serial.print(millis() / 1000);
    Serial.println(" seconds");
    Serial.print("4kHz samples: ");
    Serial.println(sample_count);
    Serial.print("100Hz samples: ");
    Serial.println(processed_100hz_count);
    Serial.print("Inferences: ");
    Serial.println(inference_count);
    Serial.print("Effective sample rate: ");
    Serial.print(sample_count / (millis() / 1000.0f), 1);
    Serial.println(" Hz");
    Serial.println("------------------");
    Serial.println();
    last_stats = millis();
  }
}
