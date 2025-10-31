#include "NotchFilterCascade.h"
#include <math.h>

NotchFilterCascade::NotchFilterCascade() {
  reset();
}

void NotchFilterCascade::reset() {
  // Clear all state variables
  x1_30 = x2_30 = y1_30 = y2_30 = 0.0f;
  x1_50 = x2_50 = y1_50 = y2_50 = 0.0f;
  x1_100 = x2_100 = y1_100 = y2_100 = 0.0f;
  x1_150 = x2_150 = y1_150 = y2_150 = 0.0f;
}

float NotchFilterCascade::applyNotch(float input, float f_n, float* x1, float* x2, float* y1, float* y2) {
  // Simple notch filter transfer function
  // H(z) = (1 - 2*cos(2*pi*f_n)*z^-1 + z^-2) / (1 - 2*r*cos(2*pi*f_n)*z^-1 + r^2*z^-2)

  float omega = 2.0f * M_PI * f_n;
  float cos_omega = cosf(omega);

  // Numerator coefficients
  float b0 = 1.0f;
  float b1 = -2.0f * cos_omega;
  float b2 = 1.0f;

  // Denominator coefficients
  float a1 = -2.0f * NOTCH_RADIUS * cos_omega;
  float a2 = NOTCH_RADIUS * NOTCH_RADIUS;

  // Apply difference equation
  float output = b0 * input + b1 * (*x1) + b2 * (*x2)
                 - a1 * (*y1) - a2 * (*y2);

  // Update state
  *x2 = *x1;
  *x1 = input;
  *y2 = *y1;
  *y1 = output;

  return output;
}

float NotchFilterCascade::process(float input) {
  // Apply notch filters in cascade (matching old implementation line 710)
  // Order: 50Hz (filter1), 100Hz (filter2), 150Hz (filter3), then 30Hz (filter4)

  float output = input;

  // 50Hz fundamental
  output = applyNotch(output, F_N_50, &x1_50, &x2_50, &y1_50, &y2_50);

  // 100Hz 2nd harmonic
  output = applyNotch(output, F_N_100, &x1_100, &x2_100, &y1_100, &y2_100);

  // 150Hz 3rd harmonic
  output = applyNotch(output, F_N_150, &x1_150, &x2_150, &y1_150, &y2_150);

  // 30Hz notch
  output = applyNotch(output, F_N_30, &x1_30, &x2_30, &y1_30, &y2_30);

  return output;
}
