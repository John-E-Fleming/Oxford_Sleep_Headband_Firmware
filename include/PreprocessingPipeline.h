#pragma once

#include "BetaBandpassFilter.h"
#include "NotchFilterCascade.h"

// Complete preprocessing pipeline matching colleague's implementation
// Pipeline: 4000Hz raw → 500Hz → BP filter → Notch filters → 100Hz
class PreprocessingPipeline {
public:
  PreprocessingPipeline();
  void reset();

  // Process one sample at 4000Hz input rate
  // Returns true when a new 100Hz sample is ready, false otherwise
  // When true, the output parameter contains the processed sample
  bool processSample(float input_4000hz, float& output_100hz);

private:
  BetaBandpassFilter betaFilter_;
  NotchFilterCascade notchFilter_;

  // Downsampling buffers
  float downsample_500hz_buffer_[8];  // For averaging every 8 samples (4000Hz→500Hz)
  int downsample_500hz_count_;

  float downsample_100hz_buffer_[5];  // For averaging every 5 samples (500Hz→100Hz)
  int downsample_100hz_count_;
};
