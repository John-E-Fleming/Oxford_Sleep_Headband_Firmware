#pragma once

#include <Arduino.h>
#include "model.h"
#include <SdFat.h>

// Simple streaming logger for sleep stage predictions
// Writes epoch number, timestamp, probabilities, and predicted stage to CSV
class InferenceLogger {
public:
  InferenceLogger();

  // Initialize logger - opens file for streaming writes
  bool begin(const String& filename = "");

  // Log a prediction
  // epoch_number: 0-indexed epoch count
  // epoch_end_seconds: recording timestamp at end of epoch (epoch 0 = 30s, etc.)
  // probabilities: array of 5 floats [Wake, N1, N2, N3, REM]
  void logPrediction(int epoch_number,
                     float epoch_end_seconds,
                     const float* probabilities);

  // Get record count
  size_t getRecordCount() const { return record_count_; }

  // Close the log file
  void close();

  // Check if logging is active
  bool isLogging() const { return log_file_.isOpen(); }

  // Print summary to Serial
  void printSummary() const;

private:
  SdFile log_file_;
  bool initialized_;
  size_t record_count_;

  // Stage counts for summary
  int stage_counts_[5];  // [Wake, N1, N2, N3, REM]

  // Helper to get stage name from index
  const char* getStageName(int stage_index) const;
};
