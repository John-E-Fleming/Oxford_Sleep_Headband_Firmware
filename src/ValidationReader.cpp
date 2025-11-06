/*
 * ValidationReader.cpp
 *
 * Implementation of validation against reference predictions
 */

#include "ValidationReader.h"
#include <string.h>
#include <ctype.h>

ValidationReader::ValidationReader()
  : predictions_(nullptr), num_epochs_(0), num_compared_(0), num_matches_(0), total_mse_(0.0f) {
  // Memory allocation deferred to begin() method
}

ValidationReader::~ValidationReader() {
  if (predictions_) {
    extmem_free(predictions_);
    predictions_ = nullptr;
  }
}

bool ValidationReader::begin(const char* filename, SdFat* sd) {
  // Allocate predictions array in external RAM (PSRAM) if not already allocated
  if (!predictions_) {
    predictions_ = (ReferencePrediction*)extmem_malloc(
        sizeof(ReferencePrediction) * MAX_VALIDATION_EPOCHS);

    if (!predictions_) {
      Serial.println("ERROR: Failed to allocate memory for validation predictions");
      return false;
    }
  }

  if (!sd) {
    Serial.println("ERROR: SD card not initialized");
    return false;
  }

  Serial.print("Loading reference predictions from: ");
  Serial.println(filename);

  SdFile file;
  if (!file.open(filename, O_RDONLY)) {
    Serial.println("ERROR: Failed to open reference predictions file");
    return false;
  }

  char line[256];
  int line_num = 0;
  num_epochs_ = 0;

  // Read CSV file line by line
  while (file.available() && num_epochs_ < MAX_VALIDATION_EPOCHS) {
    int n = file.fgets(line, sizeof(line));
    if (n <= 0) break;

    line_num++;

    // Skip header line
    if (line_num == 1) {
      continue;
    }

    // Remove trailing newline/carriage return
    int len = strlen(line);
    while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) {
      line[--len] = '\0';
    }

    // Skip empty lines
    if (len == 0) continue;

    // Parse CSV line
    if (parseCsvLine(line, predictions_[num_epochs_])) {
      num_epochs_++;
    } else {
      Serial.print("WARNING: Failed to parse line ");
      Serial.println(line_num);
    }
  }

  file.close();

  Serial.print("Loaded ");
  Serial.print(num_epochs_);
  Serial.println(" reference predictions");

  if (num_epochs_ >= MAX_VALIDATION_EPOCHS) {
    Serial.println("WARNING: Reached maximum epoch limit");
  }

  return num_epochs_ > 0;
}

bool ValidationReader::parseCsvLine(const char* line, ReferencePrediction& pred) {
  // Expected format: Epoch,Predicted_Stage,Wake,N1,N2,N3,REM
  // Example: 0,Wake,0.91796875,0.05859375,0.01953125,0.0,0.0

  char buffer[256];
  strncpy(buffer, line, sizeof(buffer) - 1);
  buffer[sizeof(buffer) - 1] = '\0';

  // Parse epoch number
  char* token = strtok(buffer, ",");
  if (!token) return false;
  pred.epoch = atoi(token);

  // Parse predicted stage
  token = strtok(NULL, ",");
  if (!token) return false;
  strncpy(pred.predicted_stage, token, sizeof(pred.predicted_stage) - 1);
  pred.predicted_stage[sizeof(pred.predicted_stage) - 1] = '\0';

  // Parse probabilities (Wake, N1, N2, N3, REM)
  for (int i = 0; i < 5; i++) {
    token = strtok(NULL, ",");
    if (!token) return false;
    pred.probabilities[i] = atof(token);
  }

  return true;
}

int ValidationReader::findEpochIndex(int epoch_num) const {
  // Linear search (fast enough for validation purposes)
  for (int i = 0; i < num_epochs_; i++) {
    if (predictions_[i].epoch == epoch_num) {
      return i;
    }
  }
  return -1;
}

float ValidationReader::calculateMSE(const float* probs1, const float* probs2) const {
  float sum = 0.0f;
  for (int i = 0; i < 5; i++) {
    float diff = probs1[i] - probs2[i];
    sum += diff * diff;
  }
  return sum / 5.0f;  // Mean squared error
}

bool ValidationReader::stagesMatch(const char* stage1, const char* stage2) const {
  // Case-insensitive comparison to handle "WAKE" vs "Wake", etc.
  return strcasecmp(stage1, stage2) == 0;
}

bool ValidationReader::compareAndLog(int epoch_num,
                                     const float* measured_probs,
                                     const char* measured_stage) {
  // Find reference prediction for this epoch
  int ref_idx = findEpochIndex(epoch_num);
  if (ref_idx < 0) {
    Serial.print("WARNING: No reference for epoch ");
    Serial.println(epoch_num);
    return false;
  }

  const ReferencePrediction& ref = predictions_[ref_idx];

  // Calculate MSE between probability distributions
  float mse = calculateMSE(measured_probs, ref.probabilities);
  total_mse_ += mse;

  // Check if stages match
  bool match = stagesMatch(measured_stage, ref.predicted_stage);
  num_compared_++;
  if (match) {
    num_matches_++;
  }

  // Log mismatch details
  if (!match) {
    Serial.print("MISMATCH Epoch ");
    Serial.print(epoch_num);
    Serial.print(": Teensy=");
    Serial.print(measured_stage);
    Serial.print(", Reference=");
    Serial.print(ref.predicted_stage);
    Serial.print(" | MSE=");
    Serial.println(mse, 6);
  }

  // Log periodic summary (every 10 epochs)
  if (num_compared_ % 10 == 0) {
    float agreement = getAgreementPercent();
    float mean_mse = getMeanMSE();

    Serial.print("Epoch ");
    Serial.print(epoch_num);
    Serial.print(" | Agreement: ");
    Serial.print(num_matches_);
    Serial.print("/");
    Serial.print(num_compared_);
    Serial.print(" (");
    Serial.print(agreement, 1);
    Serial.print("%) | Mean MSE: ");
    Serial.println(mean_mse, 6);
  }

  return true;
}

float ValidationReader::getAgreementPercent() const {
  if (num_compared_ == 0) return 0.0f;
  return 100.0f * num_matches_ / num_compared_;
}

float ValidationReader::getMeanMSE() const {
  if (num_compared_ == 0) return 0.0f;
  return total_mse_ / num_compared_;
}

void ValidationReader::printSummary() const {
  Serial.println();
  Serial.println("========================================");
  Serial.println("VALIDATION SUMMARY");
  Serial.println("========================================");
  Serial.print("Total epochs compared: ");
  Serial.println(num_compared_);
  Serial.print("Exact stage matches: ");
  Serial.print(num_matches_);
  Serial.print("/");
  Serial.print(num_compared_);
  Serial.print(" (");
  Serial.print(getAgreementPercent(), 1);
  Serial.println("%)");
  Serial.print("Mean probability MSE: ");
  Serial.println(getMeanMSE(), 6);

  if (num_compared_ > 0) {
    int mismatches = num_compared_ - num_matches_;
    Serial.print("Mismatches: ");
    Serial.print(mismatches);
    Serial.println(" (logged to serial above)");
  }

  Serial.println("========================================");
  Serial.println();
}
