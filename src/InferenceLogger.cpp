#include "InferenceLogger.h"
#include <SdFat.h>

extern SdFat sd; // Use existing SD card instance

InferenceLogger::InferenceLogger() : initialized_(false) {
}

void InferenceLogger::begin(size_t initial_capacity) {
  records_.reserve(initial_capacity);
  initialized_ = true;
  Serial.println("Inference logger initialized");
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
  record.snr_estimate = 0.0f; // Will be set if quality metrics available
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
  record.signal_min = 0.0f; // Could be calculated if needed
  record.signal_max = 0.0f;
  
  // Processing parameters
  record.window_size_seconds = window_seconds;
  record.slide_interval_seconds = slide_seconds;
  
  records_.push_back(record);
}

void InferenceLogger::logQualityRejection(unsigned long timestamp_ms,
                                         unsigned long sample_start,
                                         unsigned long sample_end,
                                         int epoch_number,
                                         const EEGQualityChecker::QualityMetrics& quality,
                                         float signal_mean,
                                         float signal_std) {
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
  record.rejection_reason = quality.rejection_reason;
  record.snr_estimate = quality.snr_estimate;
  record.artifact_percentage = quality.artifact_percentage;
  record.outlier_count = quality.outlier_count;
  
  // No model outputs (inference not performed)
  record.predicted_stage = WAKE; // Default
  record.max_confidence = 0.0f;
  for (int i = 0; i < 5; i++) {
    record.confidence_scores[i] = 0.0f;
  }
  
  // Signal statistics
  record.signal_mean = signal_mean;
  record.signal_std = signal_std;
  record.signal_min = 0.0f;
  record.signal_max = 0.0f;
  
  records_.push_back(record);
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

String InferenceLogger::exportToCSV(bool include_header) const {
  String csv = "";
  
  if (include_header) {
    csv += getCSVHeader() + "\n";
  }
  
  for (const auto& record : records_) {
    csv += recordToCSV(record) + "\n";
  }
  
  return csv;
}

bool InferenceLogger::saveToFile(const String& filename) {
  if (!sd.exists("/inference_logs")) {
    sd.mkdir("/inference_logs");
  }
  
  String filepath = "/inference_logs/" + filename;
  SdFile file;
  
  if (!file.open(filepath.c_str(), O_WRITE | O_CREAT | O_TRUNC)) {
    Serial.println("Failed to open log file for writing");
    return false;
  }
  
  // Write header
  file.println(getCSVHeader());
  
  // Write records
  for (const auto& record : records_) {
    file.println(recordToCSV(record));
  }
  
  file.close();
  Serial.print("Inference log saved to: ");
  Serial.println(filepath);
  return true;
}

InferenceLogger::SummaryStats InferenceLogger::getSummary() const {
  SummaryStats stats = {0};
  
  stats.total_attempts = records_.size();
  
  float total_confidence = 0.0f;
  float total_snr = 0.0f;
  float total_artifacts = 0.0f;
  int quality_checks = 0;
  
  for (const auto& record : records_) {
    if (record.quality_passed) {
      stats.successful_inferences++;
      total_confidence += record.max_confidence;
      
      // Count sleep stages
      switch (record.predicted_stage) {
        case WAKE: stats.wake_count++; break;
        case LIGHT_SLEEP: stats.n1_count++; break;
        case DEEP_SLEEP: stats.n2_count++; break;
        case 3: stats.n3_count++; break;
        case REM_SLEEP: stats.rem_count++; break;
      }
    } else {
      stats.quality_rejections++;
    }
    
    if (record.snr_estimate > 0) {
      total_snr += record.snr_estimate;
      quality_checks++;
    }
    
    total_artifacts += record.artifact_percentage;
  }
  
  if (stats.total_attempts > 0) {
    stats.quality_pass_rate = 100.0f * stats.successful_inferences / stats.total_attempts;
    stats.mean_artifact_rate = total_artifacts / stats.total_attempts;
  }
  
  if (stats.successful_inferences > 0) {
    stats.mean_confidence = total_confidence / stats.successful_inferences;
  }
  
  if (quality_checks > 0) {
    stats.mean_snr = total_snr / quality_checks;
  }
  
  return stats;
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
    case WAKE: return "WAKE";
    case LIGHT_SLEEP: return "N1";
    case DEEP_SLEEP: return "N2";
    case 3: return "N3";
    case REM_SLEEP: return "REM";
    default: return "UNKNOWN";
  }
}

std::vector<InferenceLogger::InferenceRecord> InferenceLogger::findRecordsByTime(
    float start_seconds, float end_seconds) const {
  std::vector<InferenceRecord> filtered;
  
  for (const auto& record : records_) {
    if (record.time_seconds >= start_seconds && record.time_seconds <= end_seconds) {
      filtered.push_back(record);
    }
  }
  
  return filtered;
}

std::vector<std::pair<int, int>> InferenceLogger::findStageTransitions() const {
  std::vector<std::pair<int, int>> transitions;
  
  if (records_.size() < 2) return transitions;
  
  for (size_t i = 1; i < records_.size(); i++) {
    if (records_[i].inference_performed && records_[i-1].inference_performed) {
      if (records_[i].predicted_stage != records_[i-1].predicted_stage) {
        transitions.push_back({i-1, i});
      }
    }
  }
  
  return transitions;
}

void InferenceLogger::clear() {
  records_.clear();
}