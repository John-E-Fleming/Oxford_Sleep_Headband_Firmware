#pragma once

// Simple FIR notch filter cascade matching colleague's implementation
// Designed for 500Hz sample rate
// Filters out 50Hz, 100Hz, 150Hz (power line harmonics) and 30Hz
class NotchFilterCascade {
public:
  NotchFilterCascade();
  void reset();
  float process(float input);

private:
  // Simple FIR notch filter implementation
  // For a simple notch at normalized frequency f_n, the transfer function is:
  // H(z) = (1 - 2*cos(2*pi*f_n)*z^-1 + z^-2) / (1 - 2*r*cos(2*pi*f_n)*z^-1 + r^2*z^-2)
  // where r is close to 1 (e.g., 0.95) for a narrow notch

  static constexpr float SAMPLE_RATE = 500.0f;
  static constexpr float NOTCH_RADIUS = 0.95f;  // Controls notch width

  // Notch frequencies
  static constexpr float F_30HZ = 30.0f;
  static constexpr float F_50HZ = 50.0f;
  static constexpr float F_100HZ = 100.0f;
  static constexpr float F_150HZ = 150.0f;

  // Normalized frequencies (f_n = 2 * f / fs)
  static constexpr float F_N_30 = 2.0f * F_30HZ / SAMPLE_RATE;
  static constexpr float F_N_50 = 2.0f * F_50HZ / SAMPLE_RATE;
  static constexpr float F_N_100 = 2.0f * F_100HZ / SAMPLE_RATE;
  static constexpr float F_N_150 = 2.0f * F_150HZ / SAMPLE_RATE;

  // Helper function to apply single notch filter
  float applyNotch(float input, float f_n, float* x1, float* x2, float* y1, float* y2);

  // State variables for each notch filter
  float x1_30, x2_30, y1_30, y2_30;    // 30Hz notch
  float x1_50, x2_50, y1_50, y2_50;    // 50Hz fundamental
  float x1_100, x2_100, y1_100, y2_100; // 100Hz 2nd harmonic
  float x1_150, x2_150, y1_150, y2_150; // 150Hz 3rd harmonic
};
