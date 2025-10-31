#pragma once

// 6th order IIR bandpass filter matching colleague's implementation
// Designed for 500Hz sample rate
// Implemented as cascade of 3 second-order sections (SOS) for numerical stability
class BetaBandpassFilter {
public:
  BetaBandpassFilter();
  void reset();
  float process(float input);

private:
  // Filter coefficients from old implementation (lines 494-500)
  // 6th order IIR bandpass filter at 500Hz sample rate
  static constexpr float beta_gk[3] = {
    0.220763911414562363777491782457218505442f,
    0.220763911414562363777491782457218505442f,
    0.202176685305338443843226059470907784998f
  };

  // Numerator coefficients [b0, b1, b2] for each section
  static constexpr float beta_bk0[3][3] = {
    {1.0f, 0.0f, -1.0f},
    {1.0f, 0.0f, -1.0f},
    {1.0f, 0.0f, -1.0f}
  };

  // Denominator coefficients [a1, a2] for each section (a0=1)
  static constexpr float beta_ak0[3][2] = {
    {-1.993775865592432472439554658194538205862f, 0.993815691293846836806835653987945988774f},
    {-1.421117261953496413440234391600824892521f, 0.619182697104341372984492863906780257821f},
    {-1.593074522250444990945084100530948489904f, 0.59564662938932300129124541854253038764f}
  };

  // State variables for 3 second-order sections
  float xkv[3][3];  // Input history [section][sample]
  float ykv[3][3];  // Output history [section][sample]
};
