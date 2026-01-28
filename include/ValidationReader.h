/*
 * ValidationReader.h
 *
 * Validates Teensy model predictions against reference predictions from Python
 * Loads CSV file from SD card and compares epoch-by-epoch
 */

#ifndef VALIDATION_READER_H
#define VALIDATION_READER_H

#include <Arduino.h>
#include <SdFat.h>

// Maximum number of epochs to store (adjust based on available memory)
#define MAX_VALIDATION_EPOCHS 2000

// Reference prediction data structure
struct ReferencePrediction {
  int epoch;
  char predicted_stage[8];  // "Wake", "N1", "N2", "N3", "REM"
  float probabilities[5];   // Wake, N1, N2, N3, REM
};

class ValidationReader {
public:
  ValidationReader();
  ~ValidationReader();

  // Initialize and load reference predictions from CSV file
  // Returns true on success, false on failure
  bool begin(const char* filename, SdFat* sd);

  // Check if reference predictions are loaded
  bool isLoaded() const { return num_epochs_ > 0; }

  // Get number of loaded epochs
  int getNumEpochs() const { return num_epochs_; }

  // Compare current prediction against reference
  // Returns true if epoch exists in reference data
  bool compareAndLog(int epoch_num,
                     const float* measured_probs,
                     const char* measured_stage);

  // Get current agreement statistics
  float getAgreementPercent() const;
  int getNumMatches() const { return num_matches_; }
  int getNumCompared() const { return num_compared_; }
  float getMeanMSE() const;

  // Print summary statistics
  void printSummary();

  // Enable saving Teensy predictions to CSV file
  bool enablePredictionLogging(const char* output_filename);
  void closePredictionLog();

private:
  // Output file for Teensy predictions
  SdFile output_file_;
  bool logging_enabled_;
  SdFat* sd_ptr_;
  // Storage for reference predictions (allocated in external RAM)
  ReferencePrediction* predictions_;
  int num_epochs_;

  // Statistics tracking
  int num_compared_;
  int num_matches_;
  float total_mse_;

  // Helper functions
  bool parseCsvLine(const char* line, ReferencePrediction& pred);
  float calculateMSE(const float* probs1, const float* probs2) const;
  bool stagesMatch(const char* stage1, const char* stage2) const;
  int findEpochIndex(int epoch_num) const;
};

#endif // VALIDATION_READER_H
