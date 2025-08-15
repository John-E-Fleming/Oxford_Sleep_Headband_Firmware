#pragma once

// Simple 4th order Butterworth bandpass filter (0.5-40Hz at 100Hz sample rate)
// Implemented as cascade of 2nd order sections for numerical stability
class BandpassFilter {
public:
  BandpassFilter();
  void reset();
  float process(float input);

private:
  // Filter coefficients for 0.5-40Hz bandpass at 100Hz sample rate
  // Calculated offline using scipy.signal.butter(4, [0.5, 40], 'bandpass', fs=100, output='sos')
  
  // Section 1 coefficients (b0, b1, b2, a0, a1, a2)
  static constexpr float b0_1 = 0.00482434;
  static constexpr float b1_1 = 0.00000000;
  static constexpr float b2_1 = -0.00482434;
  static constexpr float a0_1 = 1.00000000;
  static constexpr float a1_1 = -1.67782457;
  static constexpr float a2_1 = 0.72087509;
  
  // Section 2 coefficients
  static constexpr float b0_2 = 1.00000000;
  static constexpr float b1_2 = 0.00000000;
  static constexpr float b2_2 = -1.00000000;
  static constexpr float a0_2 = 1.00000000;
  static constexpr float a1_2 = -1.56112372;
  static constexpr float a2_2 = 0.64135154;
  
  // State variables for section 1
  float x1_1, x2_1;  // input history
  float y1_1, y2_1;  // output history
  
  // State variables for section 2
  float x1_2, x2_2;  // input history
  float y1_2, y2_2;  // output history
};