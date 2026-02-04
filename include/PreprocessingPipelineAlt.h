#pragma once

#include "BandpassFilter100Hz.h"

// Alternative preprocessing pipeline for testing different downsampling approaches
// Pipeline options:
//   A/B: 4kHz -> 500Hz -> 100Hz -> Bandpass filter at 100Hz
//   C/D: 4kHz -> 100Hz (direct) -> Bandpass filter at 100Hz
//
// Key difference from main pipeline: filter applied AFTER resampling to 100Hz
class PreprocessingPipelineAlt {
public:
  // Downsampling method options
  enum DownsampleMethod {
    DECIMATE,         // Option A: 4kHz->500Hz(decimate)->100Hz(avg)->filter
    AVERAGE,          // Option B: 4kHz->500Hz(average)->100Hz(avg)->filter
    DIRECT_DECIMATE,  // Option C: 4kHz->100Hz(decimate, every 40th)->filter
    DIRECT_AVERAGE    // Option D: 4kHz->100Hz(average 40 samples)->filter
  };

  PreprocessingPipelineAlt();

  // Reset all internal state
  void reset();

  // Set the downsampling method (default: AVERAGE)
  void setDownsampleMethod(DownsampleMethod method);

  // Process a single 4kHz sample
  // Returns true when a 100Hz output sample is ready
  bool processSample(float input_4000hz, float& output_100hz);

  // Get current downsampling method
  DownsampleMethod getDownsampleMethod() const { return method_; }

private:
  DownsampleMethod method_;

  // For two-stage downsampling (Options A, B): 4kHz -> 500Hz -> 100Hz
  // Stage 1: 4kHz -> 500Hz (8:1 ratio)
  float downsample_500hz_buffer_[8];
  int downsample_500hz_count_;

  // Stage 2: 500Hz -> 100Hz (5:1 ratio, always averaging)
  float downsample_100hz_buffer_[5];
  int downsample_100hz_count_;

  // For direct downsampling (Options C, D): 4kHz -> 100Hz (40:1 ratio)
  float direct_100hz_buffer_[40];
  int direct_100hz_count_;

  // Bandpass filter at 100Hz (used by all methods)
  BandpassFilter100Hz filter_;
};
