#include "BandpassFilter.h"

BandpassFilter::BandpassFilter() {
  reset();
}

void BandpassFilter::reset() {
  // Clear all state variables
  x1_1 = x2_1 = 0.0f;
  y1_1 = y2_1 = 0.0f;
  x1_2 = x2_2 = 0.0f;
  y1_2 = y2_2 = 0.0f;
}

float BandpassFilter::process(float input) {
  // Section 1: Apply first 2nd order section
  float output_1 = b0_1 * input + b1_1 * x1_1 + b2_1 * x2_1
                   - a1_1 * y1_1 - a2_1 * y2_1;
  
  // Update section 1 state
  x2_1 = x1_1;
  x1_1 = input;
  y2_1 = y1_1;
  y1_1 = output_1;
  
  // Section 2: Apply second 2nd order section
  float output_2 = b0_2 * output_1 + b1_2 * x1_2 + b2_2 * x2_2
                   - a1_2 * y1_2 - a2_2 * y2_2;
  
  // Update section 2 state
  x2_2 = x1_2;
  x1_2 = output_1;
  y2_2 = y1_2;
  y1_2 = output_2;
  
  return output_2;
}