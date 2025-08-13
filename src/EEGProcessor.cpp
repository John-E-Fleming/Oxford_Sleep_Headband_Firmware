#include "EEGProcessor.h"
#include <math.h>

EEGProcessor::EEGProcessor() : stats_initialized_(false) {
  // Initialize channel statistics
  for (int i = 0; i < ADS1299_CHANNELS; i++) {
    channel_means_[i] = 0.0f;
    channel_stds_[i] = 1.0f;  // Start with unit variance
  }
}

bool EEGProcessor::begin() {
  // Initialize ring buffer (already done in constructor)
  Serial.println("EEG Processor initialized");
  return true;
}

void EEGProcessor::addSample(float* channels) {
  // Add all channels for this sample to ring buffer
  for (int i = 0; i < ADS1299_CHANNELS; i++) {
    if (!sample_buffer_.push(channels[i])) {
      // Buffer full, this will overwrite oldest data automatically in ring buffer
      Serial.println("Sample buffer overflow");
    }
  }
  
  // Update running statistics for normalization
  updateChannelStats(channels);
}

bool EEGProcessor::isWindowReady() {
  // Check if we have enough samples for a smaller window (reduced for testing)
  return sample_buffer_.size() >= 1000;  // Much smaller window for memory constraints
}

bool EEGProcessor::getProcessedWindow(float* output_buffer) {
  if (!isWindowReady()) {
    return false;
  }
  
  // Use a much smaller window for memory constraints
  const int small_window_size = 1000;
  
  // Get samples from ring buffer (most recent window)
  int buffer_size = sample_buffer_.size();
  int start_index = buffer_size - small_window_size;
  if (start_index < 0) start_index = 0;
  
  for (int i = 0; i < small_window_size && i < MODEL_INPUT_SIZE; i++) {
    output_buffer[i] = sample_buffer_[start_index + i];
  }
  
  return true;
}

void EEGProcessor::preprocessData(float* raw_data, float* processed_data, int length) {
  // Copy data first
  memcpy(processed_data, raw_data, length * sizeof(float));
  
  // Apply channel-wise normalization
  for (int sample = 0; sample < WINDOW_SIZE_SAMPLES; sample++) {
    for (int ch = 0; ch < ADS1299_CHANNELS; ch++) {
      int index = sample * ADS1299_CHANNELS + ch;
      
      if (stats_initialized_) {
        // Z-score normalization: (x - mean) / std
        processed_data[index] = (processed_data[index] - channel_means_[ch]) / channel_stds_[ch];
      }
      
      // Clamp to reasonable range to prevent extreme values
      if (processed_data[index] > 5.0f) processed_data[index] = 5.0f;
      if (processed_data[index] < -5.0f) processed_data[index] = -5.0f;
    }
  }
  
  // Additional preprocessing can be added here:
  // - Bandpass filtering (0.5-35 Hz typical for sleep EEG)
  // - Artifact rejection
  // - Feature extraction (spectral features, etc.)
}

void EEGProcessor::updateChannelStats(float* sample) {
  const float learning_rate = 0.001f;  // Slow adaptation
  
  if (!stats_initialized_) {
    // Initialize with first sample
    for (int i = 0; i < ADS1299_CHANNELS; i++) {
      channel_means_[i] = sample[i];
    }
    stats_initialized_ = true;
    return;
  }
  
  // Running average update
  for (int i = 0; i < ADS1299_CHANNELS; i++) {
    // Update mean
    float old_mean = channel_means_[i];
    channel_means_[i] = old_mean + learning_rate * (sample[i] - old_mean);
    
    // Update standard deviation (simplified running std)
    float deviation = sample[i] - channel_means_[i];
    channel_stds_[i] = channel_stds_[i] + learning_rate * (abs(deviation) - channel_stds_[i]);
    
    // Ensure std doesn't get too small
    if (channel_stds_[i] < 0.1f) {
      channel_stds_[i] = 0.1f;
    }
  }
}

void EEGProcessor::normalizeChannel(float* data, int channel, int length) {
  if (!stats_initialized_) return;
  
  for (int i = channel; i < length; i += ADS1299_CHANNELS) {
    data[i] = (data[i] - channel_means_[channel]) / channel_stds_[channel];
  }
}

float EEGProcessor::applyLowPassFilter(float new_sample, float prev_filtered, float alpha) {
  // Simple exponential moving average low-pass filter
  return alpha * new_sample + (1.0f - alpha) * prev_filtered;
}