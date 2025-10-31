#include "BetaBandpassFilter.h"
#include <stdint.h>

// Define static constexpr members (required for older C++ standards)
constexpr float BetaBandpassFilter::beta_gk[3];
constexpr float BetaBandpassFilter::beta_bk0[3][3];
constexpr float BetaBandpassFilter::beta_ak0[3][2];

BetaBandpassFilter::BetaBandpassFilter() {
  reset();
}

void BetaBandpassFilter::reset() {
  // Clear all state variables
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      xkv[i][j] = 0.0f;
      ykv[i][j] = 0.0f;
    }
  }
}

float BetaBandpassFilter::process(float input) {
  // Implementation matches old code lines 368-383
  float filtered_data = input;

  for (uint8_t i = 0; i < 3; i++) {
    // Shift past input samples in input buffer xkv
    xkv[i][0] = xkv[i][1];
    xkv[i][1] = xkv[i][2];
    xkv[i][2] = filtered_data * beta_gk[i];

    // Shift past output samples in output buffer ykv
    ykv[i][0] = ykv[i][1];
    ykv[i][1] = ykv[i][2];

    // Apply difference equation
    ykv[i][2] = (beta_bk0[i][2] * xkv[i][0] + beta_bk0[i][1] * xkv[i][1] + beta_bk0[i][0] * xkv[i][2])
                - (beta_ak0[i][1] * ykv[i][0] + beta_ak0[i][0] * ykv[i][1]);

    filtered_data = ykv[i][2];
  }

  return filtered_data;
}
