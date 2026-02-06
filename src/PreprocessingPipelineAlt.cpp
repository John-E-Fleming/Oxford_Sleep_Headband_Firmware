#include "PreprocessingPipelineAlt.h"

PreprocessingPipelineAlt::PreprocessingPipelineAlt(int input_sample_rate)
    : method_(DIRECT_AVERAGE),  // Default to Option D (best performing, works with any rate)
      input_sample_rate_(input_sample_rate),
      downsample_ratio_(input_sample_rate / 100),
      downsample_500hz_count_(0),
      downsample_100hz_count_(0),
      direct_100hz_count_(0) {
  reset();
}

void PreprocessingPipelineAlt::reset() {
  // Reset 4kHz -> 500Hz buffer (for Options A, B)
  for (int i = 0; i < 8; i++) {
    downsample_500hz_buffer_[i] = 0.0f;
  }
  downsample_500hz_count_ = 0;

  // Reset 500Hz -> 100Hz buffer (for Options A, B)
  for (int i = 0; i < 5; i++) {
    downsample_100hz_buffer_[i] = 0.0f;
  }
  downsample_100hz_count_ = 0;

  // Reset direct input -> 100Hz buffer (for Options C, D)
  for (int i = 0; i < MAX_DOWNSAMPLE_RATIO; i++) {
    direct_100hz_buffer_[i] = 0.0f;
  }
  direct_100hz_count_ = 0;

  // Reset filter
  filter_.reset();
}

void PreprocessingPipelineAlt::setDownsampleMethod(DownsampleMethod method) {
  // Options A and B (two-stage: 4kHz->500Hz->100Hz) only work with 4kHz input
  // For other sample rates, fall back to DIRECT_AVERAGE (Option D)
  if ((method == DECIMATE || method == AVERAGE) && input_sample_rate_ != 4000) {
    method_ = DIRECT_AVERAGE;
  } else {
    method_ = method;
  }
}

bool PreprocessingPipelineAlt::processSample(float input, float& output_100hz) {
  // ===========================================================================
  // Direct downsampling path (Options C, D): input_rate -> 100Hz
  // Uses dynamic downsample_ratio_ based on input sample rate
  // ===========================================================================
  if (method_ == DIRECT_DECIMATE || method_ == DIRECT_AVERAGE) {
    direct_100hz_buffer_[direct_100hz_count_] = input;
    direct_100hz_count_++;

    if (direct_100hz_count_ < downsample_ratio_) {
      return false;  // Not enough samples yet
    }

    // We have downsample_ratio_ samples - compute 100Hz output based on method
    float sample_100hz;
    if (method_ == DIRECT_DECIMATE) {
      // Option C: Take first sample (simple decimation)
      sample_100hz = direct_100hz_buffer_[0];
    } else {
      // Option D: Average all samples (acts as low-pass anti-aliasing filter)
      sample_100hz = 0.0f;
      for (int i = 0; i < downsample_ratio_; i++) {
        sample_100hz += direct_100hz_buffer_[i];
      }
      sample_100hz /= (float)downsample_ratio_;
    }

    // Reset counter for next group
    direct_100hz_count_ = 0;

    // Apply bandpass filter at 100Hz
    output_100hz = filter_.process(sample_100hz);
    return true;  // 100Hz sample ready
  }

  // ===========================================================================
  // Two-stage downsampling path (Options A, B): 4kHz -> 500Hz -> 100Hz
  // NOTE: This path only works for 4kHz input. For other sample rates,
  // setDownsampleMethod() will have redirected to DIRECT_AVERAGE.
  // ===========================================================================

  // Stage 1: 4kHz -> 500Hz (8:1 ratio)
  downsample_500hz_buffer_[downsample_500hz_count_] = input;
  downsample_500hz_count_++;

  if (downsample_500hz_count_ < 8) {
    return false;  // Not enough samples yet
  }

  // We have 8 samples - compute 500Hz output based on method
  float sample_500hz;
  if (method_ == DECIMATE) {
    // Option A: Take every 8th sample (simple decimation)
    sample_500hz = downsample_500hz_buffer_[0];
  } else {
    // Option B: Average 8 samples (acts as low-pass anti-aliasing filter)
    sample_500hz = 0.0f;
    for (int i = 0; i < 8; i++) {
      sample_500hz += downsample_500hz_buffer_[i];
    }
    sample_500hz /= 8.0f;
  }

  // Reset counter for next group
  downsample_500hz_count_ = 0;

  // Stage 2: 500Hz -> 100Hz (5:1 ratio, always averaging)
  downsample_100hz_buffer_[downsample_100hz_count_] = sample_500hz;
  downsample_100hz_count_++;

  if (downsample_100hz_count_ < 5) {
    return false;  // Not enough samples yet
  }

  // We have 5 samples at 500Hz - average them for 100Hz output
  float sample_100hz = 0.0f;
  for (int i = 0; i < 5; i++) {
    sample_100hz += downsample_100hz_buffer_[i];
  }
  sample_100hz /= 5.0f;

  // Reset counter for next group
  downsample_100hz_count_ = 0;

  // Stage 3: Bandpass filter at 100Hz (0.5-30Hz, 5th order Butterworth)
  output_100hz = filter_.process(sample_100hz);

  return true;  // 100Hz sample ready
}
