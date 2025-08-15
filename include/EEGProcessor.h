#pragma once

#include <Arduino.h>
#include "model.h"

// EEG configuration for ML inference
#define ADS1299_CHANNELS 9           // 9 channels as per MATLAB code
#define ML_SAMPLE_RATE 100           // 100 Hz for ML processing 
#define ML_WINDOW_SIZE_SECONDS 30    // 30-second windows for ML model
#define ML_WINDOW_SIZE_SAMPLES (ML_SAMPLE_RATE * ML_WINDOW_SIZE_SECONDS)  // 3000 samples

// Simple circular buffer for Teensy compatibility
template<typename T, size_t N>
class CircularBuffer {
public:
  CircularBuffer() : head_(0), tail_(0), count_(0) {}
  
  bool push(const T& item) {
    if (count_ >= N) {
      // Buffer full, overwrite oldest
      tail_ = (tail_ + 1) % N;
    } else {
      count_++;
    }
    buffer_[head_] = item;
    head_ = (head_ + 1) % N;
    return true;
  }
  
  size_t size() const { return count_; }
  
  T operator[](size_t index) const {
    if (index >= count_) return T{};
    size_t pos = (tail_ + index) % N;
    return buffer_[pos];
  }
  
private:
  T buffer_[N];
  size_t head_;
  size_t tail_;
  size_t count_;
};

class EEGProcessor {
public:
  EEGProcessor();
  
  // Initialize the processor
  bool begin();
  
  // Add new EEG sample (called from ADS1299 interrupt)
  void addSample(float* channels);
  
  // Add single filtered sample for ML processing
  void addFilteredSample(float sample);
  
  // Check if enough data is available for inference
  bool isWindowReady();
  
  // Get processed window for ML inference
  bool getProcessedWindow(float* output_buffer);
  
  // Apply preprocessing (filtering, normalization, etc.)
  void preprocessData(float* raw_data, float* processed_data, int length);
  
  // Get normalization statistics for debugging
  float getFilteredMean() const { return filtered_mean_; }
  float getFilteredStd() const { return filtered_std_; }

private:
  // Circular buffer for multi-channel raw data (legacy)
  CircularBuffer<float, 2000> sample_buffer_;  // Fixed size: ~8KB for floats
  
  // Circular buffer for single-channel filtered data (for ML inference)
  CircularBuffer<float, ML_WINDOW_SIZE_SAMPLES> filtered_buffer_;  // 3000 samples = ~12KB
  
  // Preprocessing parameters
  float filtered_mean_;
  float filtered_std_;
  bool stats_initialized_;
  
  // Legacy multi-channel stats (kept for compatibility)
  float channel_means_[ADS1299_CHANNELS];
  float channel_stds_[ADS1299_CHANNELS];
  
  // Helper functions
  void updateChannelStats(float* sample);
  void normalizeChannel(float* data, int channel, int length);
  float applyLowPassFilter(float new_sample, float prev_filtered, float alpha);
};