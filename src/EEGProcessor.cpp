#include "EEGProcessor.h"
#include <math.h>

EEGProcessor::EEGProcessor() : filtered_mean_(0.0f), filtered_std_(1.0f), stats_initialized_(false) {
  // Initialize channel statistics arrays
  for (int i = 0; i < ADS1299_CHANNELS; i++) {
    channel_means_[i] = 0.0f;
    channel_stds_[i] = 1.0f;
  }
}

bool EEGProcessor::begin() {
  // Initialize ring buffer (already done in constructor)
  Serial.println("EEG Processor initialized");
  return true;
}

void EEGProcessor::addSample(float* channels) {
  // Add all channels for this sample to ring buffer (legacy support)
  for (int i = 0; i < ADS1299_CHANNELS; i++) {
    if (!sample_buffer_.push(channels[i])) {
      // Buffer full, this will overwrite oldest data automatically in ring buffer
      Serial.println("Sample buffer overflow");
    }
  }
}

void EEGProcessor::addFilteredSample(float sample) {
  // Add single filtered sample to ML buffer
  filtered_buffer_.push(sample);
  
  // Update running statistics for normalization
  const float learning_rate = 0.001f;
  
  if (!stats_initialized_) {
    filtered_mean_ = sample;
    stats_initialized_ = true;
    return;
  }
  
  // Update mean and std for filtered data
  float old_mean = filtered_mean_;
  filtered_mean_ = old_mean + learning_rate * (sample - old_mean);
  
  float deviation = sample - filtered_mean_;
  filtered_std_ = filtered_std_ + learning_rate * (abs(deviation) - filtered_std_);
  
  // Ensure std doesn't get too small
  if (filtered_std_ < 0.1f) {
    filtered_std_ = 0.1f;
  }
}

bool EEGProcessor::isWindowReady() {
  // Check if we have enough samples for ML inference (30 seconds = 3000 samples at 100Hz)
  return filtered_buffer_.size() >= ML_WINDOW_SIZE_SAMPLES;
}

bool EEGProcessor::getProcessedWindow(float* output_buffer) {
  if (!isWindowReady()) {
    return false;
  }
  
  // Get the most recent 30-second window from filtered buffer
  int buffer_size = filtered_buffer_.size();
  int start_index = buffer_size - ML_WINDOW_SIZE_SAMPLES;
  if (start_index < 0) start_index = 0;
  
  // Copy and normalize the window
  for (int i = 0; i < ML_WINDOW_SIZE_SAMPLES && i < MODEL_INPUT_SIZE; i++) {
    float raw_sample = filtered_buffer_[start_index + i];
    
    if (stats_initialized_) {
      // Z-score normalization: (x - mean) / std
      output_buffer[i] = (raw_sample - filtered_mean_) / filtered_std_;
      
      // Clamp to reasonable range to prevent extreme values
      if (output_buffer[i] > 5.0f) output_buffer[i] = 5.0f;
      if (output_buffer[i] < -5.0f) output_buffer[i] = -5.0f;
    } else {
      output_buffer[i] = raw_sample;
    }
  }
  
  return true;
}

void EEGProcessor::preprocessData(float* raw_data, float* processed_data, int length) {
  // Copy data first
  memcpy(processed_data, raw_data, length * sizeof(float));
  
  // Apply normalization to the provided length
  for (int i = 0; i < length; i++) {
    if (stats_initialized_) {
      // For legacy compatibility, assume single channel normalization
      processed_data[i] = (processed_data[i] - filtered_mean_) / filtered_std_;
      
      // Clamp to reasonable range to prevent extreme values
      if (processed_data[i] > 5.0f) processed_data[i] = 5.0f;
      if (processed_data[i] < -5.0f) processed_data[i] = -5.0f;
    }
  }
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