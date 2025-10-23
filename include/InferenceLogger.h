#pragma once

#include <Arduino.h>
#include "model.h"
#include "EEGQualityChecker.h"
#include <SdFat.h>

// Streaming logging system for ML inference results and quality metrics
// Writes directly to SD card to minimize memory usage
class InferenceLogger {
public:
  // Lightweight record structure for immediate writing
  struct InferenceRecord {
    // Timing information
    unsigned long timestamp_ms;        // Milliseconds since start
    float time_seconds;                // Seconds since start
    unsigned long sample_index_start;  // First sample index in window
    unsigned long sample_index_end;    // Last sample index in window
    int epoch_number;                  // Inference epoch counter
    
    // Quality metrics
    bool quality_passed;               // Whether quality check passed
    float snr_estimate;                // Signal-to-noise ratio
    float artifact_percentage;         // Percentage of artifacts in window
    int outlier_count;                 // Number of outliers detected
    const char* rejection_reason;      // If rejected, why? (pointer to avoid String allocation)
    
    // Model outputs
    bool inference_performed;          // Whether inference ran (vs quality fail)
    SleepStage predicted_stage;        // Predicted sleep stage
    float confidence_scores[5];        // Raw model outputs [Wake, N1, N2, N3, REM]
    float max_confidence;              // Highest confidence score
    
    // Signal statistics (for debugging)
    float signal_mean;                 // Mean of processed window
    float signal_std;                  // Std deviation of processed window
    
    // Processing parameters (for reproducibility)
    int window_size_seconds;           // Window duration
    int slide_interval_seconds;        // Sliding interval
  };
  
  InferenceLogger();
  
  // Initialize logger with streaming to SD card
  bool begin(const String& filename = "");
  
  // Log a successful inference
  void logInference(unsigned long timestamp_ms, 
                   unsigned long sample_start,
                   unsigned long sample_end,
                   int epoch_number,
                   const float* confidence_scores,
                   SleepStage predicted_stage,
                   float signal_mean,
                   float signal_std,
                   int window_seconds = 30,
                   int slide_seconds = 10);
  
  // Log a quality rejection
  void logQualityRejection(unsigned long timestamp_ms,
                          unsigned long sample_start,
                          unsigned long sample_end,
                          int epoch_number,
                          const EEGQualityChecker::QualityMetrics& quality,
                          float signal_mean,
                          float signal_std);
  
  // Get record count
  size_t getRecordCount() const { return record_count_; }
  
  // Get CSV header
  String getCSVHeader() const;
  
  // Close the log file
  void close();
  
  // Check if logging is active
  bool isLogging() const { return log_file_.isOpen(); }
  
  // Get summary statistics (real-time counters)
  struct SummaryStats {
    int total_attempts;
    int successful_inferences;
    int quality_rejections;
    float quality_pass_rate;
    float mean_confidence;
    int wake_count;
    int n1_count;
    int n2_count;
    int n3_count;
    int rem_count;
    float mean_snr;
    float mean_artifact_rate;
  };
  
  SummaryStats getSummary() const;
  
  // Print summary to Serial
  void printSummary() const;

private:
  SdFile log_file_;
  bool initialized_;
  size_t record_count_;
  
  // Real-time statistics tracking (no memory storage)
  SummaryStats stats_;
  float total_confidence_sum_;
  float total_snr_sum_;
  float total_artifacts_sum_;
  int quality_checks_count_;
  
  // Helper functions
  String getStageName(SleepStage stage) const;
  String recordToCSV(const InferenceRecord& record) const;
  void updateStats(const InferenceRecord& record);
};