#pragma once

#include <Arduino.h>

// EEG signal quality assessment for artifact detection
// Prevents inference on corrupted/out-of-distribution data
class EEGQualityChecker {
public:
  // Quality assessment results
  struct QualityMetrics {
    bool is_valid;                // Overall pass/fail
    bool amplitude_ok;             // Amplitude within physiological range
    bool noise_ok;                 // Not too noisy
    bool flatline_ok;              // Not flatlined/disconnected
    bool clipping_ok;              // Not clipping/saturated
    float snr_estimate;            // Signal-to-noise ratio estimate
    float artifact_percentage;     // Percentage of samples flagged as artifacts
    int outlier_count;             // Number of extreme outliers
    String rejection_reason;       // Human-readable rejection reason
  };

  EEGQualityChecker();
  
  // Configure thresholds (can be loaded from config)
  void setThresholds(float max_amplitude_uv = 500.0f,    // Max physiological amplitude (microvolts)
                     float min_amplitude_uv = 0.5f,       // Min to avoid flatline
                     float max_std_dev = 100.0f,          // Max standard deviation
                     float min_std_dev = 0.1f,            // Min to avoid flatline
                     float outlier_z_score = 4.0f,        // Z-score for outlier detection
                     float max_artifact_percent = 20.0f); // Max % of bad samples allowed
  
  // Check window quality (after filtering/normalization)
  QualityMetrics checkWindowQuality(const float* data, int num_samples, 
                                   float sample_mean, float sample_std);
  
  // Check raw signal quality (before filtering)
  QualityMetrics checkRawSignalQuality(const float* data, int num_samples);
  
  // Quick check for single sample (for real-time monitoring)
  bool isSampleValid(float sample, float running_mean, float running_std);
  
  // Get current statistics
  void getStatistics(int& total_windows, int& rejected_windows, 
                     int& total_samples, int& artifact_samples);
  
  // Reset statistics
  void resetStatistics();

private:
  // Configurable thresholds
  float max_amplitude_uv_;
  float min_amplitude_uv_;
  float max_std_dev_;
  float min_std_dev_;
  float outlier_z_threshold_;
  float max_artifact_percent_;
  
  // Statistics tracking
  int total_windows_checked_;
  int windows_rejected_;
  int total_samples_checked_;
  int artifact_samples_detected_;
  
  // Helper functions
  float calculateMedian(float* data, int num_samples);
  float calculateMAD(const float* data, int num_samples, float median); // Median Absolute Deviation
  int detectOutliers(const float* data, int num_samples, float median, float mad);
  bool detectFlatline(const float* data, int num_samples);
  bool detectClipping(const float* data, int num_samples, float threshold);
  float estimateSNR(const float* data, int num_samples);
};