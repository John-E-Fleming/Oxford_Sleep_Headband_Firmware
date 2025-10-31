#include "PreprocessingPipeline.h"

PreprocessingPipeline::PreprocessingPipeline()
  : downsample_500hz_count_(0), downsample_100hz_count_(0) {
  reset();
}

void PreprocessingPipeline::reset() {
  betaFilter_.reset();
  notchFilter_.reset();

  // Clear downsampling buffers
  for (int i = 0; i < 8; i++) {
    downsample_500hz_buffer_[i] = 0.0f;
  }
  for (int i = 0; i < 5; i++) {
    downsample_100hz_buffer_[i] = 0.0f;
  }

  downsample_500hz_count_ = 0;
  downsample_100hz_count_ = 0;
}

bool PreprocessingPipeline::processSample(float input_4000hz, float& output_100hz) {
  // Stage 1: Accumulate samples for 4000Hz → 500Hz downsampling
  downsample_500hz_buffer_[downsample_500hz_count_] = input_4000hz;
  downsample_500hz_count_++;

  // Check if we have 8 samples to average (4000Hz → 500Hz)
  if (downsample_500hz_count_ < 8) {
    return false;  // Need more samples
  }

  // Average every 8 samples to get 500Hz
  float sum_500hz = 0.0f;
  for (int i = 0; i < 8; i++) {
    sum_500hz += downsample_500hz_buffer_[i];
  }
  float sample_500hz = sum_500hz / 8.0f;
  downsample_500hz_count_ = 0;  // Reset for next group

  // Stage 2: Apply beta bandpass filter at 500Hz
  float filtered_500hz = betaFilter_.process(sample_500hz);

  // Stage 3: Apply notch filters at 500Hz
  float notched_500hz = notchFilter_.process(filtered_500hz);

  // Stage 4: Accumulate samples for 500Hz → 100Hz downsampling
  downsample_100hz_buffer_[downsample_100hz_count_] = notched_500hz;
  downsample_100hz_count_++;

  // Check if we have 5 samples to average (500Hz → 100Hz)
  if (downsample_100hz_count_ < 5) {
    return false;  // Need more samples
  }

  // Average every 5 samples to get 100Hz (matching old code lines 772-773)
  float sum_100hz = 0.0f;
  for (int i = 0; i < 5; i++) {
    sum_100hz += downsample_100hz_buffer_[i];
  }
  output_100hz = sum_100hz / 5.0f;
  downsample_100hz_count_ = 0;  // Reset for next group

  return true;  // New 100Hz sample ready
}
