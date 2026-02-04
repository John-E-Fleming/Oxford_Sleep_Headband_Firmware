#include "PreprocessingPipelineAlt.h"

PreprocessingPipelineAlt::PreprocessingPipelineAlt()
    : method_(AVERAGE), downsample_500hz_count_(0), downsample_100hz_count_(0), direct_100hz_count_(0) {
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

  // Reset direct 4kHz -> 100Hz buffer (for Options C, D)
  for (int i = 0; i < 40; i++) {
    direct_100hz_buffer_[i] = 0.0f;
  }
  direct_100hz_count_ = 0;

  // Reset filter
  filter_.reset();
}

void PreprocessingPipelineAlt::setDownsampleMethod(DownsampleMethod method) {
  method_ = method;
}

bool PreprocessingPipelineAlt::processSample(float input_4000hz, float& output_100hz) {
  // ===========================================================================
  // Direct downsampling path (Options C, D): 4kHz -> 100Hz (40:1 ratio)
  // ===========================================================================
  if (method_ == DIRECT_DECIMATE || method_ == DIRECT_AVERAGE) {
    direct_100hz_buffer_[direct_100hz_count_] = input_4000hz;
    direct_100hz_count_++;

    if (direct_100hz_count_ < 40) {
      return false;  // Not enough samples yet
    }

    // We have 40 samples - compute 100Hz output based on method
    float sample_100hz;
    if (method_ == DIRECT_DECIMATE) {
      // Option C: Take every 40th sample (simple decimation)
      sample_100hz = direct_100hz_buffer_[0];
    } else {
      // Option D: Average 40 samples (acts as low-pass anti-aliasing filter)
      sample_100hz = 0.0f;
      for (int i = 0; i < 40; i++) {
        sample_100hz += direct_100hz_buffer_[i];
      }
      sample_100hz /= 40.0f;
    }

    // Reset counter for next group
    direct_100hz_count_ = 0;

    // Apply bandpass filter at 100Hz
    output_100hz = filter_.process(sample_100hz);
    return true;  // 100Hz sample ready
  }

  // ===========================================================================
  // Two-stage downsampling path (Options A, B): 4kHz -> 500Hz -> 100Hz
  // ===========================================================================

  // Stage 1: 4kHz -> 500Hz (8:1 ratio)
  downsample_500hz_buffer_[downsample_500hz_count_] = input_4000hz;
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
