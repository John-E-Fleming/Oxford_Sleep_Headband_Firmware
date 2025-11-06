#include "PreprocessingPipeline.h"

PreprocessingPipeline::PreprocessingPipeline()
  : downsample_250hz_count_(0), resample_count_(0), output_phase_(0) {
  reset();
}

void PreprocessingPipeline::reset() {
  trainingFilter_.reset();

  // Clear downsampling buffers
  for (int i = 0; i < 16; i++) {
    downsample_250hz_buffer_[i] = 0.0f;
  }
  for (int i = 0; i < 5; i++) {
    resample_buffer_[i] = 0.0f;
  }

  downsample_250hz_count_ = 0;
  resample_count_ = 0;
  output_phase_ = 0;
}

bool PreprocessingPipeline::processSample(float input_4000hz, float& output_100hz) {
  /*
   * Preprocessing pipeline matching training script:
   * 1. Downsample 4000Hz → 250Hz (average every 16 samples)
   * 2. Apply Butterworth bandpass filter (0.5-30 Hz) at 250Hz
   * 3. Resample 250Hz → 100Hz (5:2 rational resampling)
   */

  // Stage 1: Accumulate samples for 4000Hz → 250Hz downsampling
  downsample_250hz_buffer_[downsample_250hz_count_] = input_4000hz;
  downsample_250hz_count_++;

  // Check if we have 16 samples to average (4000Hz → 250Hz)
  if (downsample_250hz_count_ < 16) {
    return false;  // Need more samples
  }

  // Average every 16 samples to get 250Hz
  float sum_250hz = 0.0f;
  for (int i = 0; i < 16; i++) {
    sum_250hz += downsample_250hz_buffer_[i];
  }
  float sample_250hz = sum_250hz / 16.0f;
  downsample_250hz_count_ = 0;  // Reset for next group

  // Stage 2: Apply training Butterworth bandpass filter (0.5-30 Hz) at 250Hz
  float filtered_250hz = trainingFilter_.process(sample_250hz);

  // Stage 3: Fractional resampling 250Hz → 100Hz (5:2 ratio)
  // We collect 5 samples at 250Hz and output 2 samples at 100Hz
  // This gives us exactly 100Hz: 5 samples / 250 Hz = 0.02 seconds = 2 samples / 100 Hz

  resample_buffer_[resample_count_] = filtered_250hz;
  resample_count_++;

  // Check if we have 5 samples for resampling
  if (resample_count_ < 5) {
    return false;  // Need more samples
  }

  // Use linear interpolation to generate 2 output samples from 5 input samples
  // Output sample 0 is at position 0.0 (first input sample)
  // Output sample 1 is at position 2.5 (between 2nd and 3rd input sample)

  if (output_phase_ == 0) {
    // First output: use first sample directly (position 0.0)
    output_100hz = resample_buffer_[0];
    output_phase_ = 1;
    return true;  // Output first of two samples
  } else {
    // Second output: interpolate at position 2.5
    // This is halfway between sample 2 and sample 3 (0-indexed)
    output_100hz = (resample_buffer_[2] + resample_buffer_[3]) * 0.5f;

    // Reset for next group of 5 samples
    resample_count_ = 0;
    output_phase_ = 0;
    return true;  // Output second of two samples
  }
}
