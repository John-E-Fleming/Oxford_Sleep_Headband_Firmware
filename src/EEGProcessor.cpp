#include "EEGProcessor.h"
#include <math.h>

EEGProcessor::EEGProcessor() : 
    window_size_samples_(ML_WINDOW_SIZE_SAMPLES),
    inference_interval_samples_(ML_INFERENCE_INTERVAL_SAMPLES),
    samples_since_last_inference_(0),
    first_window_ready_(false),
    filtered_mean_(0.0f), 
    filtered_std_(1.0f), 
    stats_initialized_(false) {
  // Initialize channel statistics arrays
  for (int i = 0; i < ADS1299_CHANNELS; i++) {
    channel_means_[i] = 0.0f;
    channel_stds_[i] = 1.0f;
  }
}

bool EEGProcessor::begin() {
  // Initialize ring buffer (already done in constructor)
  Serial.println("EEG Processor initialized");
  Serial.print("Window size: ");
  Serial.print(window_size_samples_ / ML_SAMPLE_RATE);
  Serial.println(" seconds");
  Serial.print("Inference interval: ");
  Serial.print(inference_interval_samples_ / ML_SAMPLE_RATE);
  Serial.println(" seconds");
  return true;
}

void EEGProcessor::configureSlidingWindow(int window_seconds, int inference_interval_seconds) {
  window_size_samples_ = window_seconds * ML_SAMPLE_RATE;
  inference_interval_samples_ = inference_interval_seconds * ML_SAMPLE_RATE;
  samples_since_last_inference_ = 0;
  
  Serial.print("Sliding window configured: ");
  Serial.print(window_seconds);
  Serial.print("s window, ");
  Serial.print(inference_interval_seconds);
  Serial.print("s interval (");
  Serial.print(100.0f * (window_seconds - inference_interval_seconds) / window_seconds);
  Serial.println("% overlap)");
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
  
  // Track samples for sliding window inference timing
  samples_since_last_inference_++;
  
  // Mark when we have enough samples for first window
  if (!first_window_ready_ && filtered_buffer_.size() >= static_cast<size_t>(window_size_samples_)) {
    first_window_ready_ = true;
    Serial.println("First inference window ready");
  }
  
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
  // Check if we have enough samples for ML inference
  return filtered_buffer_.size() >= static_cast<size_t>(window_size_samples_);
}

bool EEGProcessor::isInferenceTimeReady() {
  // First window: wait for full window
  if (!first_window_ready_) {
    return isWindowReady();
  }
  
  // Subsequent windows: check if we've collected enough new samples
  return samples_since_last_inference_ >= inference_interval_samples_;
}

void EEGProcessor::markInferenceComplete() {
  // Reset the sample counter for next inference interval
  samples_since_last_inference_ = 0;
}

bool EEGProcessor::getProcessedWindow(float* output_buffer) {
  if (!isWindowReady()) {
    return false;
  }
  
  // Get the most recent window from filtered buffer
  int buffer_size = filtered_buffer_.size();
  int start_index = buffer_size - window_size_samples_;
  if (start_index < 0) start_index = 0;
  
  // Copy and normalize the window
  // Now using full 3000 samples as MODEL_INPUT_SIZE was fixed
  int samples_to_copy = min(window_size_samples_, MODEL_INPUT_SIZE - 1);  // -1 for epoch index
  for (int i = 0; i < samples_to_copy; i++) {
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
  
  // Add epoch index as last element (will be set by caller)
  output_buffer[samples_to_copy] = 0.0f;  // Placeholder for epoch index
  
  return true;
}

bool EEGProcessor::getProcessedWindowInt8(int8_t* output_buffer, float scale, int32_t zero_point, int epoch_index) {
  // Get float window first
  float temp_buffer[MODEL_INPUT_SIZE];
  if (!getProcessedWindow(temp_buffer)) {
    return false;
  }
  
  // Quantize to INT8 (matching reference implementation)
  int samples_to_quantize = min(window_size_samples_, MODEL_INPUT_SIZE - 1);
  for (int i = 0; i < samples_to_quantize; i++) {
    // Quantize using model's scale and zero point
    int32_t quantized = round(temp_buffer[i] / scale) + zero_point;
    
    // Clamp to int8 range
    if (quantized > 127) quantized = 127;
    if (quantized < -128) quantized = -128;
    
    output_buffer[i] = static_cast<int8_t>(quantized);
  }
  
  // Add quantized epoch index as last element
  int32_t epoch_quantized = round(static_cast<float>(epoch_index) / scale) + zero_point;
  if (epoch_quantized > 127) epoch_quantized = 127;
  if (epoch_quantized < -128) epoch_quantized = -128;
  output_buffer[samples_to_quantize] = static_cast<int8_t>(epoch_quantized);
  
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