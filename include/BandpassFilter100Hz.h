#pragma once

// 5th order Butterworth bandpass filter (0.5-30Hz at 100Hz sample rate)
// Matches the filter parameters used in Ali's training script
// Implemented as cascade of 5 second-order sections (biquads) for numerical stability
class BandpassFilter100Hz {
public:
  BandpassFilter100Hz();
  void reset();
  float process(float input);

private:
  // Number of cascaded biquad sections
  static constexpr int NUM_SECTIONS = 5;

  // Filter coefficients for 0.5-30Hz bandpass at 100Hz sample rate
  // Generated with: scipy.signal.butter(5, [0.5, 30], 'bandpass', fs=100, output='sos')

  // Coefficients stored as [b0, b1, b2, a1, a2] for each section (a0 = 1.0 always)
  static constexpr float sos_[NUM_SECTIONS][5] = {
    // Section 0
    {0.1013716411f, 0.2027432822f, 0.1013716411f, 0.3345078535f, 0.1321522896f},
    // Section 1
    {1.0000000000f, 2.0000000000f, 1.0000000000f, 0.4768819035f, 0.5527533469f},
    // Section 2
    {1.0000000000f, 0.0000000000f, -1.0000000000f, -0.8213745394f, -0.1423210757f},
    // Section 3
    {1.0000000000f, -2.0000000000f, 1.0000000000f, -1.9491215378f, 0.9501129365f},
    // Section 4
    {1.0000000000f, -2.0000000000f, 1.0000000000f, -1.9801445789f, 0.9811262501f}
  };

  // State variables for each biquad section (Direct Form II)
  float w1_[NUM_SECTIONS];  // w[n-1] delay
  float w2_[NUM_SECTIONS];  // w[n-2] delay
};
