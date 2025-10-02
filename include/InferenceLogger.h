#pragma once

#include <Arduino.h>
#include <vector>
#include "model.h"
#include "EEGQualityChecker.h"

// Comprehensive logging system for ML inference results and quality metrics
// Enables post-hoc analysis of model performance against recorded EEG data
class InferenceLogger {
public:
  // Complete record of each inference attempt
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
    String rejection_reason;           // If rejected, why?
    
    // Model outputs
    bool inference_performed;          // Whether inference ran (vs quality fail)
    SleepStage predicted_stage;        // Predicted sleep stage
    float confidence_scores[5];        // Raw model outputs [Wake, N1, N2, N3, REM]
    float max_confidence;              // Highest confidence score
    
    // Signal statistics (for debugging)
    float signal_mean;                 // Mean of processed window
    float signal_std;                  // Std deviation of processed window
    float signal_min;                  // Min value in window
    float signal_max;                  // Max value in window
    
    // Processing parameters (for reproducibility)
    int window_size_seconds;           // Window duration
    int slide_interval_seconds;        // Sliding interval
  };
  
  InferenceLogger();
  
  // Initialize logger with optional capacity pre-allocation
  void begin(size_t initial_capacity = 1000);
  
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
  
  // Get records for analysis
  const std::vector<InferenceRecord>& getRecords() const { return records_; }
  size_t getRecordCount() const { return records_.size(); }
  
  // Export to CSV format (returns as String for Serial or SD writing)
  String exportToCSV(bool include_header = true) const;
  
  // Export single record as CSV line
  String recordToCSV(const InferenceRecord& record) const;
  
  // Get CSV header
  String getCSVHeader() const;
  
  // Write to SD card file
  bool saveToFile(const String& filename);
  
  // Clear all records
  void clear();
  
  // Get summary statistics
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
  
  // Find records by time range (for correlation with external events)
  std::vector<InferenceRecord> findRecordsByTime(float start_seconds, float end_seconds) const;
  
  // Find transitions between sleep stages
  std::vector<std::pair<int, int>> findStageTransitions() const;

private:
  std::vector<InferenceRecord> records_;
  bool initialized_;
  
  // Helper to get stage name
  String getStageName(SleepStage stage) const;
};