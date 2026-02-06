#pragma once

#include "BandpassFilter100Hz.h"

// Alternative preprocessing pipeline for testing different downsampling approaches
// Pipeline options:
//   A/B: 4kHz -> 500Hz -> 100Hz -> Bandpass filter at 100Hz (4kHz only)
//   C/D: input_rate -> 100Hz (direct) -> Bandpass filter at 100Hz (any rate)
//
// Key difference from main pipeline: filter applied AFTER resampling to 100Hz
// Supports configurable input sample rates (1000Hz, 2000Hz, 4000Hz, etc.)
class PreprocessingPipelineAlt {
public:
  // Downsampling method options
  enum DownsampleMethod {
    DECIMATE,         // Option A: 4kHz->500Hz(decimate)->100Hz(avg)->filter (4kHz only)
    AVERAGE,          // Option B: 4kHz->500Hz(average)->100Hz(avg)->filter (4kHz only)
    DIRECT_DECIMATE,  // Option C: input->100Hz(decimate)->filter (any rate)
    DIRECT_AVERAGE    // Option D: input->100Hz(average)->filter (any rate)
  };

  // Constructor takes input sample rate (default 4000 for backwards compatibility)
  // Sample rate must be divisible by 100
  PreprocessingPipelineAlt(int input_sample_rate = 4000);

  // Reset all internal state
  void reset();

  // Set the downsampling method (default: AVERAGE)
  // Note: Options A and B require 4kHz input; they will fall back to DIRECT_AVERAGE for other rates
  void setDownsampleMethod(DownsampleMethod method);

  // Process a single input sample
  // Returns true when a 100Hz output sample is ready
  bool processSample(float input, float& output_100hz);

  // Get current downsampling method
  DownsampleMethod getDownsampleMethod() const { return method_; }

  // Get input sample rate
  int getInputSampleRate() const { return input_sample_rate_; }

  // Get downsample ratio (input_rate / 100)
  int getDownsampleRatio() const { return downsample_ratio_; }

private:
  DownsampleMethod method_;
  int input_sample_rate_;      // Input sample rate (1000, 2000, 4000, etc.)
  int downsample_ratio_;       // Calculated: input_rate / 100

  // For two-stage downsampling (Options A, B): 4kHz -> 500Hz -> 100Hz
  // Only valid when input_sample_rate_ == 4000
  float downsample_500hz_buffer_[8];
  int downsample_500hz_count_;

  // Stage 2: 500Hz -> 100Hz (5:1 ratio, always averaging)
  float downsample_100hz_buffer_[5];
  int downsample_100hz_count_;

  // For direct downsampling (Options C, D): input_rate -> 100Hz
  // Buffer size = max ratio (40 for 4kHz)
  static const int MAX_DOWNSAMPLE_RATIO = 40;
  float direct_100hz_buffer_[MAX_DOWNSAMPLE_RATIO];
  int direct_100hz_count_;

  // Bandpass filter at 100Hz (used by all methods)
  BandpassFilter100Hz filter_;
};
