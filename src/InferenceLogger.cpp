#include "InferenceLogger.h"
#include <SdFat.h>

extern SdFat sd; // Use existing SD card instance

InferenceLogger::InferenceLogger() 
  : initialized_(false), record_count_(0), total_confidence_sum_(0), 
    total_snr_sum_(0), total_artifacts_sum_(0), quality_checks_count_(0) {
  // Initialize stats
  stats_ = {0};
}

bool InferenceLogger::begin(const String& filename) {
  if (initialized_) {
    return true;
  }
  
  String log_filename = filename;
  if (log_filename.length() == 0) {
    log_filename = "inference_" + String(millis()) + ".csv";
  }
  
  // Create logs directory if it doesn't exist
  if (!sd.exists("/inference_logs")) {
    sd.mkdir("/inference_logs");
  }
  
  String filepath = "/inference_logs/" + log_filename;
  
  if (!log_file_.open(filepath.c_str(), O_WRITE | O_CREAT | O_TRUNC)) {
    Serial.println("Failed to open log file for streaming writes");
    return false;
  }
  
  // Write CSV header immediately
  log_file_.println(getCSVHeader());
  log_file_.sync();  // Use sync() instead of flush() for SdFile
  
  initialized_ = true;
  record_count_ = 0;
  
  Serial.print("Streaming inference logger initialized: ");
  Serial.println(filepath);
  return true;
}

void InferenceLogger::logInference(unsigned long timestamp_ms, 
                                  unsigned long sample_start,
                                  unsigned long sample_end,
                                  int epoch_number,
                                  const float* confidence_scores,
                                  SleepStage predicted_stage,
                                  float signal_mean,
                                  float signal_std,
                                  int window_seconds,
                                  int slide_seconds) {
  if (!initialized_ || !log_file_.isOpen()) {
    return;
  }
  
  InferenceRecord record;
  
  // Timing information
  record.timestamp_ms = timestamp_ms;
  record.time_seconds = timestamp_ms / 1000.0f;
  record.sample_index_start = sample_start;
  record.sample_index_end = sample_end;
  record.epoch_number = epoch_number;
  
  // Quality metrics (inference was performed, so quality passed)
  record.quality_passed = true;
  record.inference_performed = true;
  record.rejection_reason = "";
  record.snr_estimate = 0.0f;
  record.artifact_percentage = 0.0f;
  record.outlier_count = 0;
  
  // Model outputs
  record.predicted_stage = predicted_stage;
  record.max_confidence = 0.0f;
  for (int i = 0; i < 5; i++) {
    record.confidence_scores[i] = confidence_scores[i];
    if (confidence_scores[i] > record.max_confidence) {
      record.max_confidence = confidence_scores[i];
    }
  }
  
  // Signal statistics
  record.signal_mean = signal_mean;
  record.signal_std = signal_std;
  
  // Processing parameters
  record.window_size_seconds = window_seconds;
  record.slide_interval_seconds = slide_seconds;
  
  // Write to file immediately
  log_file_.println(recordToCSV(record));
  log_file_.sync(); // Ensure data is written to SD card
  
  // Update statistics
  updateStats(record);
  record_count_++;
}

void InferenceLogger::logQualityRejection(unsigned long timestamp_ms,
                                         unsigned long sample_start,
                                         unsigned long sample_end,
                                         int epoch_number,
                                         const EEGQualityChecker::QualityMetrics& quality,
                                         float signal_mean,
                                         float signal_std) {
  if (!initialized_ || !log_file_.isOpen()) {
    return;
  }
  
  InferenceRecord record;
  
  // Timing information
  record.timestamp_ms = timestamp_ms;
  record.time_seconds = timestamp_ms / 1000.0f;
  record.sample_index_start = sample_start;
  record.sample_index_end = sample_end;
  record.epoch_number = epoch_number;
  
  // Quality metrics
  record.quality_passed = false;
  record.inference_performed = false;
  record.rejection_reason = quality.rejection_reason.c_str();
  record.snr_estimate = quality.snr_estimate;
  record.artifact_percentage = quality.artifact_percentage;
  record.outlier_count = quality.outlier_count;
  
  // No model outputs (inference not performed)
  record.predicted_stage = WAKE; // Default (yy4)
  record.max_confidence = 0.0f;
  for (int i = 0; i < 5; i++) {
    record.confidence_scores[i] = 0.0f;
  }
  
  // Signal statistics
  record.signal_mean = signal_mean;
  record.signal_std = signal_std;
  
  // Write to file immediately
  log_file_.println(recordToCSV(record));
  log_file_.sync(); // Ensure data is written to SD card
  
  // Update statistics
  updateStats(record);
  record_count_++;
}

String InferenceLogger::getCSVHeader() const {
  return "timestamp_ms,time_s,sample_start,sample_end,epoch,quality_pass,inference_run,"
         "predicted_stage,confidence_wake,confidence_n1,confidence_n2,confidence_n3,confidence_rem,"
         "max_confidence,snr,artifact_pct,outliers,signal_mean,signal_std,rejection_reason";
}

String InferenceLogger::recordToCSV(const InferenceRecord& record) const {
  String csv = "";
  csv += String(record.timestamp_ms) + ",";
  csv += String(record.time_seconds, 3) + ",";
  csv += String(record.sample_index_start) + ",";
  csv += String(record.sample_index_end) + ",";
  csv += String(record.epoch_number) + ",";
  csv += String(record.quality_passed ? 1 : 0) + ",";
  csv += String(record.inference_performed ? 1 : 0) + ",";
  csv += getStageName(record.predicted_stage) + ",";
  
  // Confidence scores
  for (int i = 0; i < 5; i++) {
    csv += String(record.confidence_scores[i], 4) + ",";
  }
  
  csv += String(record.max_confidence, 4) + ",";
  csv += String(record.snr_estimate, 2) + ",";
  csv += String(record.artifact_percentage, 2) + ",";
  csv += String(record.outlier_count) + ",";
  csv += String(record.signal_mean, 4) + ",";
  csv += String(record.signal_std, 4) + ",";
  csv += record.rejection_reason;
  
  return csv;
}

void InferenceLogger::close() {
  if (log_file_.isOpen()) {
    log_file_.close();
    Serial.println("Inference log file closed");
  }
}

InferenceLogger::SummaryStats InferenceLogger::getSummary() const {
  SummaryStats current_stats = stats_;
  
  // Calculate derived statistics
  if (current_stats.total_attempts > 0) {
    current_stats.quality_pass_rate = 100.0f * current_stats.successful_inferences / current_stats.total_attempts;
    current_stats.mean_artifact_rate = total_artifacts_sum_ / current_stats.total_attempts;
  }
  
  if (current_stats.successful_inferences > 0) {
    current_stats.mean_confidence = total_confidence_sum_ / current_stats.successful_inferences;
  }
  
  if (quality_checks_count_ > 0) {
    current_stats.mean_snr = total_snr_sum_ / quality_checks_count_;
  }
  
  return current_stats;
}

void InferenceLogger::updateStats(const InferenceRecord& record) {
  stats_.total_attempts++;
  
  if (record.quality_passed) {
    stats_.successful_inferences++;
    total_confidence_sum_ += record.max_confidence;
    
    // Count sleep stages using corrected mapping
    switch (record.predicted_stage) {
      case WAKE: stats_.wake_count++; break;            // yy0 = Wake
      case N1_VERY_LIGHT: stats_.n1_count++; break;    // yy1 = N1
      case N2_LIGHT_SLEEP: stats_.n2_count++; break;   // yy2 = N2
      case N3_DEEP_SLEEP: stats_.n3_count++; break;    // yy3 = N3
      case REM_SLEEP: stats_.rem_count++; break;        // yy4 = REM
    }
  } else {
    stats_.quality_rejections++;
  }
  
  if (record.snr_estimate > 0) {
    total_snr_sum_ += record.snr_estimate;
    quality_checks_count_++;
  }
  
  total_artifacts_sum_ += record.artifact_percentage;
}

void InferenceLogger::printSummary() const {
  auto stats = getSummary();
  
  Serial.println("\n=== Inference Log Summary ===");
  Serial.print("Total inference attempts: ");
  Serial.println(stats.total_attempts);
  Serial.print("Successful inferences: ");
  Serial.print(stats.successful_inferences);
  Serial.print(" (");
  Serial.print(stats.quality_pass_rate, 1);
  Serial.println("%)");
  Serial.print("Quality rejections: ");
  Serial.println(stats.quality_rejections);
  
  if (stats.successful_inferences > 0) {
    Serial.println("\nSleep stage distribution:");
    Serial.print("  Wake: ");
    Serial.print(stats.wake_count);
    Serial.print(" (");
    Serial.print(100.0f * stats.wake_count / stats.successful_inferences, 1);
    Serial.println("%)");
    Serial.print("  N1: ");
    Serial.print(stats.n1_count);
    Serial.print(" (");
    Serial.print(100.0f * stats.n1_count / stats.successful_inferences, 1);
    Serial.println("%)");
    Serial.print("  N2: ");
    Serial.print(stats.n2_count);
    Serial.print(" (");
    Serial.print(100.0f * stats.n2_count / stats.successful_inferences, 1);
    Serial.println("%)");
    Serial.print("  N3: ");
    Serial.print(stats.n3_count);
    Serial.print(" (");
    Serial.print(100.0f * stats.n3_count / stats.successful_inferences, 1);
    Serial.println("%)");
    Serial.print("  REM: ");
    Serial.print(stats.rem_count);
    Serial.print(" (");
    Serial.print(100.0f * stats.rem_count / stats.successful_inferences, 1);
    Serial.println("%)");
    
    Serial.print("\nMean confidence: ");
    Serial.println(stats.mean_confidence, 3);
  }
  
  Serial.print("Mean SNR: ");
  Serial.print(stats.mean_snr, 1);
  Serial.println(" dB");
  Serial.print("Mean artifact rate: ");
  Serial.print(stats.mean_artifact_rate, 1);
  Serial.println("%");
  Serial.println("=============================\n");
}

String InferenceLogger::getStageName(SleepStage stage) const {
  switch (stage) {
    case WAKE: return "WAKE";             // yy0 = Wake
    case N1_VERY_LIGHT: return "N1";      // yy1 = N1 (Very Light Sleep)
    case N2_LIGHT_SLEEP: return "N2";     // yy2 = N2 (Light Sleep)
    case N3_DEEP_SLEEP: return "N3";      // yy3 = N3 (Deep Sleep)
    case REM_SLEEP: return "REM";         // yy4 = REM Sleep
    default: return "UNKNOWN";
  }
}

