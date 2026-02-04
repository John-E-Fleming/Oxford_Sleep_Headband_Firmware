#include "BandpassFilter100Hz.h"

// Define the static constexpr coefficients
constexpr float BandpassFilter100Hz::sos_[NUM_SECTIONS][5];

BandpassFilter100Hz::BandpassFilter100Hz() {
  reset();
}

void BandpassFilter100Hz::reset() {
  for (int i = 0; i < NUM_SECTIONS; i++) {
    w1_[i] = 0.0f;
    w2_[i] = 0.0f;
  }
}

float BandpassFilter100Hz::process(float input) {
  float output = input;

  // Process through each biquad section (Direct Form II Transposed)
  for (int i = 0; i < NUM_SECTIONS; i++) {
    float b0 = sos_[i][0];
    float b1 = sos_[i][1];
    float b2 = sos_[i][2];
    float a1 = sos_[i][3];
    float a2 = sos_[i][4];

    // Direct Form II: w[n] = x[n] - a1*w[n-1] - a2*w[n-2]
    //                 y[n] = b0*w[n] + b1*w[n-1] + b2*w[n-2]
    float w0 = output - a1 * w1_[i] - a2 * w2_[i];
    output = b0 * w0 + b1 * w1_[i] + b2 * w2_[i];

    // Update state
    w2_[i] = w1_[i];
    w1_[i] = w0;
  }

  return output;
}
