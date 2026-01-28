#include "InferenceLogger.h"
#include <SdFat.h>

extern SdFat sd;  // Use existing SD card instance

InferenceLogger::InferenceLogger()
    : initialized_(false), record_count_(0) {
  for (int i = 0; i < 5; i++) {
    stage_counts_[i] = 0;
  }
}

bool InferenceLogger::begin(const String& filename) {
  if (initialized_) {
    return true;
  }

  String log_filename = filename;
  if (log_filename.length() == 0) {
    log_filename = "predictions_" + String(millis()) + ".csv";
  }

  // Create logs directory if it doesn't exist
  if (!sd.exists("/realtime_logs")) {
    sd.mkdir("/realtime_logs");
  }

  String filepath = "/realtime_logs/" + log_filename;

  if (!log_file_.open(filepath.c_str(), O_WRITE | O_CREAT | O_TRUNC)) {
    Serial.println("Failed to open prediction log file");
    return false;
  }

  // Write CSV header
  log_file_.println("epoch,timestamp_s,prob_wake,prob_n1,prob_n2,prob_n3,prob_rem,predicted_stage");
  log_file_.sync();

  initialized_ = true;
  record_count_ = 0;

  Serial.print("Prediction logger initialized: ");
  Serial.println(filepath);
  return true;
}

void InferenceLogger::logPrediction(int epoch_number,
                                    float epoch_end_seconds,
                                    const float* probabilities) {
  if (!initialized_ || !log_file_.isOpen()) {
    return;
  }

  // Find the stage with highest probability
  int max_stage = 0;
  float max_prob = probabilities[0];
  for (int i = 1; i < 5; i++) {
    if (probabilities[i] > max_prob) {
      max_prob = probabilities[i];
      max_stage = i;
    }
  }

  // Update stage counts for summary
  stage_counts_[max_stage]++;

  // Write CSV row: epoch,timestamp_s,prob_wake,prob_n1,prob_n2,prob_n3,prob_rem,predicted_stage
  log_file_.print(epoch_number);
  log_file_.print(",");
  log_file_.print(epoch_end_seconds, 1);
  log_file_.print(",");
  log_file_.print(probabilities[0], 4);  // Wake
  log_file_.print(",");
  log_file_.print(probabilities[1], 4);  // N1
  log_file_.print(",");
  log_file_.print(probabilities[2], 4);  // N2
  log_file_.print(",");
  log_file_.print(probabilities[3], 4);  // N3
  log_file_.print(",");
  log_file_.print(probabilities[4], 4);  // REM
  log_file_.print(",");
  log_file_.println(getStageName(max_stage));

  log_file_.sync();
  record_count_++;
}

void InferenceLogger::close() {
  if (log_file_.isOpen()) {
    log_file_.close();
    Serial.println("Prediction log file closed");
  }
}

void InferenceLogger::printSummary() const {
  Serial.println("\n=== Prediction Summary ===");
  Serial.print("Total epochs: ");
  Serial.println(record_count_);

  if (record_count_ > 0) {
    Serial.println("Stage distribution:");
    const char* names[] = {"Wake", "N1", "N2", "N3", "REM"};
    for (int i = 0; i < 5; i++) {
      Serial.print("  ");
      Serial.print(names[i]);
      Serial.print(": ");
      Serial.print(stage_counts_[i]);
      Serial.print(" (");
      Serial.print(100.0f * stage_counts_[i] / record_count_, 1);
      Serial.println("%)");
    }
  }
  Serial.println("==========================\n");
}

const char* InferenceLogger::getStageName(int stage_index) const {
  switch (stage_index) {
    case 0: return "Wake";
    case 1: return "N1";
    case 2: return "N2";
    case 3: return "N3";
    case 4: return "REM";
    default: return "Unknown";
  }
}
