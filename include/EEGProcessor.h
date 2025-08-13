#pragma once

#include <Arduino.h>
#include "model.h"

// EEG configuration - reduced for memory constraints
#define ADS1299_CHANNELS 9      // 9 channels as per MATLAB code
#define SAMPLE_RATE 4000        // 4000 Hz sampling rate from MATLAB
#define WINDOW_SIZE_SECONDS 1   // Reduced to 1-second windows for memory constraints
#define WINDOW_SIZE_SAMPLES (SAMPLE_RATE * WINDOW_SIZE_SECONDS)  // 4000 samples

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
  
  // Check if enough data is available for inference
  bool isWindowReady();
  
  // Get processed window for ML inference
  bool getProcessedWindow(float* output_buffer);
  
  // Apply preprocessing (filtering, normalization, etc.)
  void preprocessData(float* raw_data, float* processed_data, int length);

private:
  // Circular buffer for continuous data (much smaller for memory constraints)
  CircularBuffer<float, 2000> sample_buffer_;  // Fixed size: ~8KB for floats
  
  // Preprocessing parameters
  float channel_means_[ADS1299_CHANNELS];
  float channel_stds_[ADS1299_CHANNELS];
  bool stats_initialized_;
  
  // Helper functions
  void updateChannelStats(float* sample);
  void normalizeChannel(float* data, int channel, int length);
  float applyLowPassFilter(float new_sample, float prev_filtered, float alpha);
};