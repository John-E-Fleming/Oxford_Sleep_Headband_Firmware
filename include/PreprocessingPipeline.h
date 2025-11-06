#pragma once

#include "TrainingBandpassFilter.h"

// Complete preprocessing pipeline matching training script implementation
// Pipeline: 4000Hz raw → 250Hz → Butterworth filter → 100Hz
// This matches the exact preprocessing used during model training
class PreprocessingPipeline {
public:
  PreprocessingPipeline();
  void reset();

  // Process one sample at 4000Hz input rate
  // Returns true when a new 100Hz sample is ready, false otherwise
  // When true, the output parameter contains the processed sample
  bool processSample(float input_4000hz, float& output_100hz);

private:
  TrainingBandpassFilter trainingFilter_;

  // Downsampling buffers
  float downsample_250hz_buffer_[16];  // For averaging every 16 samples (4000Hz→250Hz)
  int downsample_250hz_count_;

  // Fractional resampler for 250Hz → 100Hz (5:2 ratio)
  // Process 5 samples at 250Hz, output 2 samples at 100Hz
  float resample_buffer_[5];  // Buffer for 5 samples at 250Hz
  int resample_count_;         // Count of samples in resample buffer
  int output_phase_;           // Which of the 2 output samples to produce next (0 or 1)
};
