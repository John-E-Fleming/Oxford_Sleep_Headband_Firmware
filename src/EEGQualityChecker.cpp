#include "EEGQualityChecker.h"
#include <algorithm>
#include <cmath>

EEGQualityChecker::EEGQualityChecker() :
    max_amplitude_uv_(500.0f),
    min_amplitude_uv_(0.5f),
    max_std_dev_(100.0f),
    min_std_dev_(0.1f),
    outlier_z_threshold_(4.0f),
    max_artifact_percent_(20.0f),
    total_windows_checked_(0),
    windows_rejected_(0),
    total_samples_checked_(0),
    artifact_samples_detected_(0) {
}

void EEGQualityChecker::setThresholds(float max_amplitude_uv, float min_amplitude_uv,
                                      float max_std_dev, float min_std_dev,
                                      float outlier_z_score, float max_artifact_percent) {
  max_amplitude_uv_ = max_amplitude_uv;
  min_amplitude_uv_ = min_amplitude_uv;
  max_std_dev_ = max_std_dev;
  min_std_dev_ = min_std_dev;
  outlier_z_threshold_ = outlier_z_score;
  max_artifact_percent_ = max_artifact_percent;
}

EEGQualityChecker::QualityMetrics EEGQualityChecker::checkWindowQuality(
    const float* data, int num_samples, float sample_mean, float sample_std) {
  
  QualityMetrics metrics;
  metrics.is_valid = true;
  metrics.amplitude_ok = true;
  metrics.noise_ok = true;
  metrics.flatline_ok = true;
  metrics.clipping_ok = true;
  metrics.outlier_count = 0;
  metrics.artifact_percentage = 0.0f;
  metrics.rejection_reason = "";
  
  total_windows_checked_++;
  total_samples_checked_ += num_samples;
  
  // Check 1: Standard deviation (catches flatlines and excessive noise)
  if (sample_std < min_std_dev_) {
    metrics.flatline_ok = false;
    metrics.is_valid = false;
    metrics.rejection_reason = "Flatline detected (std too low)";
  } else if (sample_std > max_std_dev_) {
    metrics.noise_ok = false;
    metrics.is_valid = false;
    metrics.rejection_reason = "Excessive noise (std too high)";
  }
  
  // Check 2: Count outliers and extreme values
  int artifact_count = 0;
  int clipping_count = 0;
  const float clip_threshold = 4.5f; // Z-score for clipping detection
  
  for (int i = 0; i < num_samples; i++) {
    float z_score = std::abs((data[i] - sample_mean) / sample_std);
    
    // Check for outliers
    if (z_score > outlier_z_threshold_) {
      metrics.outlier_count++;
      artifact_count++;
    }
    
    // Check for clipping (values stuck at extremes)
    if (z_score > clip_threshold) {
      clipping_count++;
    }
    
    // Check amplitude in microvolts (if data is not normalized)
    // This assumes data is already normalized, so we check z-scores instead
    if (z_score > 5.0f) {
      artifact_count++;
    }
  }
  
  // Check 3: Calculate artifact percentage
  metrics.artifact_percentage = (100.0f * artifact_count) / num_samples;
  artifact_samples_detected_ += artifact_count;
  
  if (metrics.artifact_percentage > max_artifact_percent_) {
    metrics.is_valid = false;
    metrics.amplitude_ok = false;
    if (metrics.rejection_reason.length() == 0) {
      metrics.rejection_reason = "Too many artifacts (" + 
                                 String(metrics.artifact_percentage, 1) + "%)";
    }
  }
  
  // Check 4: Detect clipping/saturation
  if (clipping_count > num_samples * 0.01f) { // More than 1% clipped
    metrics.clipping_ok = false;
    metrics.is_valid = false;
    if (metrics.rejection_reason.length() == 0) {
      metrics.rejection_reason = "Signal clipping detected";
    }
  }
  
  // Check 5: Detect repetitive patterns (might indicate disconnection)
  if (detectFlatline(data, num_samples)) {
    metrics.flatline_ok = false;
    metrics.is_valid = false;
    if (metrics.rejection_reason.length() == 0) {
      metrics.rejection_reason = "Electrode disconnection suspected";
    }
  }
  
  // Estimate SNR (simple version based on outlier ratio)
  metrics.snr_estimate = (metrics.outlier_count > 0) ? 
                         10.0f * log10f((float)num_samples / metrics.outlier_count) : 
                         40.0f; // High SNR if no outliers
  
  // Update rejection statistics
  if (!metrics.is_valid) {
    windows_rejected_++;
  }
  
  return metrics;
}

EEGQualityChecker::QualityMetrics EEGQualityChecker::checkRawSignalQuality(
    const float* data, int num_samples) {
  
  // Calculate statistics for raw data
  float sum = 0.0f;
  float sum_sq = 0.0f;
  float min_val = data[0];
  float max_val = data[0];
  
  for (int i = 0; i < num_samples; i++) {
    sum += data[i];
    sum_sq += data[i] * data[i];
    if (data[i] < min_val) min_val = data[i];
    if (data[i] > max_val) max_val = data[i];
  }
  
  float mean = sum / num_samples;
  float variance = (sum_sq / num_samples) - (mean * mean);
  float std_dev = sqrt(variance);
  
  QualityMetrics metrics;
  metrics.is_valid = true;
  metrics.amplitude_ok = true;
  metrics.noise_ok = true;
  metrics.flatline_ok = true;
  metrics.clipping_ok = true;
  
  // Check amplitude range (in microvolts)
  float amplitude_range = max_val - min_val;
  if (amplitude_range < min_amplitude_uv_) {
    metrics.flatline_ok = false;
    metrics.is_valid = false;
    metrics.rejection_reason = "Signal too weak or disconnected";
  } else if (amplitude_range > max_amplitude_uv_) {
    metrics.amplitude_ok = false;
    metrics.is_valid = false;
    metrics.rejection_reason = "Amplitude exceeds physiological range";
  }
  
  // Check for constant values (disconnection)
  if (std_dev < min_std_dev_) {
    metrics.flatline_ok = false;
    metrics.is_valid = false;
    metrics.rejection_reason = "Flatline/disconnection detected";
  }
  
  return metrics;
}

bool EEGQualityChecker::isSampleValid(float sample, float running_mean, float running_std) {
  // Quick single-sample check for real-time monitoring
  if (running_std <= 0) return true; // Can't check without statistics
  
  float z_score = std::abs((sample - running_mean) / running_std);
  return z_score < outlier_z_threshold_;
}

bool EEGQualityChecker::detectFlatline(const float* data, int num_samples) {
  // Check for consecutive identical or nearly identical values
  int consecutive_similar = 0;
  const float epsilon = 0.001f; // Tolerance for "identical" values
  const int max_consecutive = 50; // Flag if >50 consecutive similar values
  
  for (int i = 1; i < num_samples; i++) {
    if (std::abs(data[i] - data[i-1]) < epsilon) {
      consecutive_similar++;
      if (consecutive_similar > max_consecutive) {
        return true; // Flatline detected
      }
    } else {
      consecutive_similar = 0;
    }
  }
  
  return false;
}

float EEGQualityChecker::calculateMedian(float* data, int num_samples) {
  // Note: This modifies the input array due to sorting
  std::sort(data, data + num_samples);
  
  if (num_samples % 2 == 0) {
    return (data[num_samples/2 - 1] + data[num_samples/2]) / 2.0f;
  } else {
    return data[num_samples/2];
  }
}

void EEGQualityChecker::getStatistics(int& total_windows, int& rejected_windows,
                                      int& total_samples, int& artifact_samples) {
  total_windows = total_windows_checked_;
  rejected_windows = windows_rejected_;
  total_samples = total_samples_checked_;
  artifact_samples = artifact_samples_detected_;
}

void EEGQualityChecker::resetStatistics() {
  total_windows_checked_ = 0;
  windows_rejected_ = 0;
  total_samples_checked_ = 0;
  artifact_samples_detected_ = 0;
}