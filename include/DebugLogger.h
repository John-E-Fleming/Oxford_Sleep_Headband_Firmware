#pragma once

#include <Arduino.h>
#include <SdFat.h>

// Debug logger for pipeline validation
// Writes intermediate outputs to CSV files in wide format (epochs × samples)
class DebugLogger {
public:
  DebugLogger();

  // Initialize with SD card reference
  bool begin(SdFat* sd);

  // Enable/disable debug logging
  void setEnabled(bool enabled);
  bool isEnabled() const { return enabled_; }

  // Log 100Hz preprocessed data (3000 samples)
  bool logPreprocessed100Hz(float* data, int length, int epoch_index);

  // Log normalized window data (3000 samples)
  bool logNormalizedWindow(float* data, int length, int epoch_index);

  // Log quantized INT8 input (3000 samples)
  bool logQuantizedInput(int8_t* data, int length, int epoch_index);

  // Log model output probabilities (5 values: N3, N2, N1, REM, WAKE)
  bool logModelOutput(float* output, int output_size, int epoch_index, float time_seconds, const char* predicted_stage);

  // Get status
  bool isInitialized() const { return initialized_; }

private:
  SdFat* sd_;
  bool initialized_;
  bool enabled_;

  // File paths
  static const char* FILE_PREPROCESSED;
  static const char* FILE_NORMALIZED;
  static const char* FILE_QUANTIZED;
  static const char* FILE_MODEL_OUTPUT;

  // Buffer for building CSV rows (needs to be large for 3000 samples)
  static constexpr int BUFFER_SIZE = 40000;  // ~30KB + margin
  char* row_buffer_;

  // Helper functions
  bool writeHeader(const char* filename, int num_samples);
  bool appendRow(const char* filename, const char* row);
  bool fileExists(const char* filename);
};
