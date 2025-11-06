/*
 * TrainingBandpassFilter.cpp
 *
 * Implementation of 5th order Butterworth bandpass filter
 * using Direct Form II cascaded biquads for numerical stability
 */

#include "TrainingBandpassFilter.h"

// Initialize filter coefficients
// Each SOS: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]

const float TrainingBandpassFilter::b0[NUM_SECTIONS] = {
    2.579404041171073914e-03f,  // Section 1
    1.000000000000000000e+00f,  // Section 2
    1.000000000000000000e+00f,  // Section 3
    1.000000000000000000e+00f,  // Section 4
    1.000000000000000000e+00f  // Section 5
};

const float TrainingBandpassFilter::b1[NUM_SECTIONS] = {
    5.158808082342147827e-03f,  // Section 1
    2.000000000000000000e+00f,  // Section 2
    0.000000000000000000e+00f,  // Section 3
    -2.000000000000000000e+00f,  // Section 4
    -2.000000000000000000e+00f  // Section 5
};

const float TrainingBandpassFilter::b2[NUM_SECTIONS] = {
    2.579404041171073914e-03f,  // Section 1
    1.000000000000000000e+00f,  // Section 2
    -1.000000000000000000e+00f,  // Section 3
    1.000000000000000000e+00f,  // Section 4
    1.000000000000000000e+00f  // Section 5
};

const float TrainingBandpassFilter::a1[NUM_SECTIONS] = {
    -9.596837162971496582e-01f,  // Section 1
    -1.212064862251281738e+00f,  // Section 2
    -1.433070898056030273e+00f,  // Section 3
    -1.979528427124023438e+00f,  // Section 4
    -1.992304086685180664e+00f  // Section 5
};

const float TrainingBandpassFilter::a2[NUM_SECTIONS] = {
    2.994447648525238037e-01f,  // Section 1
    6.596572399139404297e-01f,  // Section 2
    4.402188658714294434e-01f,  // Section 3
    9.796913862228393555e-01f,  // Section 4
    9.924622774124145508e-01f  // Section 5
};

TrainingBandpassFilter::TrainingBandpassFilter() {
    reset();
}

void TrainingBandpassFilter::reset() {
    // Initialize all state variables to zero
    for (int i = 0; i < NUM_SECTIONS; i++) {
        w1[i] = 0.0f;
        w2[i] = 0.0f;
    }
}

float TrainingBandpassFilter::process(float input) {
    /*
     * Direct Form II implementation of cascaded biquads
     * This is the most numerically stable IIR filter structure
     *
     * For each section:
     *   w[n] = x[n] - a1*w[n-1] - a2*w[n-2]
     *   y[n] = b0*w[n] + b1*w[n-1] + b2*w[n-2]
     */

    float output = input;

    for (int i = 0; i < NUM_SECTIONS; i++) {
        // Compute current state
        float w0 = output - a1[i] * w1[i] - a2[i] * w2[i];

        // Compute output
        output = b0[i] * w0 + b1[i] * w1[i] + b2[i] * w2[i];

        // Update state
        w2[i] = w1[i];
        w1[i] = w0;
    }

    return output;
}
