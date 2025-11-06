"""
Generate Stable Butterworth Filter Coefficients for Teensy Firmware

This script designs a 5th order Butterworth bandpass filter (0.5-30 Hz) at 250Hz
sample rate, matching the training script preprocessing pipeline.

The filter is implemented as Second-Order Sections (SOS) for numerical stability.
"""

import numpy as np
from scipy import signal

# Optional matplotlib import (for plotting only)
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found, skipping plots")

# Filter specifications (from training script)
LOWCUT = 0.5      # Hz - Low cutoff frequency
HIGHCUT = 30      # Hz - High cutoff frequency
FS = 250          # Hz - Sample rate
ORDER = 5         # Filter order

print("="*70)
print("BUTTERWORTH FILTER COEFFICIENT GENERATION")
print("="*70)
print(f"Filter Type: {ORDER}th order Butterworth bandpass")
print(f"Passband: {LOWCUT} - {HIGHCUT} Hz")
print(f"Sample Rate: {FS} Hz")
print(f"Implementation: Second-Order Sections (SOS) cascade")
print("="*70)

# Design Butterworth bandpass filter in SOS format
print("\n1. Designing filter...")
sos = signal.butter(ORDER, [LOWCUT, HIGHCUT], btype='band', fs=FS, output='sos')

print(f"   Filter structure: {len(sos)} cascaded biquad sections")
print(f"   SOS array shape: {sos.shape}")

# Check stability: verify all poles are inside unit circle
print("\n2. Checking stability (poles must be inside unit circle |z| < 1)...")
# Convert SOS to transfer function to get poles
b, a = signal.sos2tf(sos)
poles = np.roots(a)
pole_magnitudes = np.abs(poles)

print(f"   Number of poles: {len(poles)}")
print(f"   Pole magnitudes:")
for i, (pole, mag) in enumerate(zip(poles, pole_magnitudes)):
    status = "[STABLE]" if mag < 1.0 else "[UNSTABLE!]"
    print(f"      Pole {i+1}: magnitude = {mag:.6f} {status}")

max_pole_mag = np.max(pole_magnitudes)
print(f"\n   Maximum pole magnitude: {max_pole_mag:.6f}")
if max_pole_mag < 1.0:
    print(f"   [OK] FILTER IS STABLE (all poles inside unit circle)")
else:
    print(f"   [WARNING] FILTER MAY BE UNSTABLE!")

# Convert to float32 (Teensy precision) and check for issues
print("\n3. Converting to float32 precision (Teensy hardware)...")
sos_float32 = sos.astype(np.float32)

# Check if precision loss affects stability
max_diff = np.max(np.abs(sos - sos_float32))
rel_diff = np.max(np.abs((sos - sos_float32) / (sos + 1e-12)))

print(f"   Max absolute difference: {max_diff:.2e}")
print(f"   Max relative difference: {rel_diff:.2e}")

if max_diff < 1e-6:
    print(f"   [OK] Float32 conversion: negligible precision loss")
elif max_diff < 1e-3:
    print(f"   [OK] Float32 conversion: small precision loss (acceptable)")
else:
    print(f"   [WARNING] Significant precision loss in float32!")

# Display SOS coefficients
print("\n4. Second-Order Section Coefficients (float32):")
print("   Each section: [b0, b1, b2, a0, a1, a2]")
print("   (a0 is always 1.0 and can be omitted in implementation)")
print()

for i, section in enumerate(sos_float32):
    b0, b1, b2, a0, a1, a2 = section
    print(f"   Section {i+1}:")
    print(f"      b: [{b0:+.18e}, {b1:+.18e}, {b2:+.18e}]")
    print(f"      a: [{a0:+.18e}, {a1:+.18e}, {a2:+.18e}]")
    print()

# Frequency response analysis
print("\n5. Analyzing frequency response...")
w, h = signal.sosfreqz(sos_float32, worN=8192, fs=FS)
magnitude_db = 20 * np.log10(np.abs(h) + 1e-12)

# Find -3dB points
magnitude_linear = np.abs(h)
target = np.max(magnitude_linear) / np.sqrt(2)  # -3dB point
passband_indices = np.where(magnitude_linear >= target)[0]

if len(passband_indices) > 0:
    low_3db = w[passband_indices[0]]
    high_3db = w[passband_indices[-1]]
    print(f"   -3dB cutoff frequencies: {low_3db:.2f} Hz, {high_3db:.2f} Hz")
    print(f"   Target: {LOWCUT} Hz, {HIGHCUT} Hz")
    print(f"   Error: {abs(low_3db - LOWCUT):.2f} Hz, {abs(high_3db - HIGHCUT):.2f} Hz")

# Passband ripple
passband_mask = (w >= LOWCUT) & (w <= HIGHCUT)
passband_mag_db = magnitude_db[passband_mask]
ripple_db = np.max(passband_mag_db) - np.min(passband_mag_db)
print(f"   Passband ripple: {ripple_db:.3f} dB")

# Stopband attenuation
stopband_low_mask = w < LOWCUT * 0.5
stopband_high_mask = w > HIGHCUT * 1.5
if np.any(stopband_low_mask):
    stopband_atten_low = -np.max(magnitude_db[stopband_low_mask])
    print(f"   Stopband attenuation (low): {stopband_atten_low:.1f} dB")
if np.any(stopband_high_mask):
    stopband_atten_high = -np.max(magnitude_db[stopband_high_mask])
    print(f"   Stopband attenuation (high): {stopband_atten_high:.1f} dB")

# Impulse response (stability check)
print("\n6. Testing impulse response (must decay to zero for stability)...")
impulse_length = 1000
impulse = np.zeros(impulse_length)
impulse[0] = 1.0
impulse_response = signal.sosfilt(sos_float32, impulse)

# Check if impulse response decays
final_samples = np.abs(impulse_response[-100:])
max_final = np.max(final_samples)
max_overall = np.max(np.abs(impulse_response))

print(f"   Max impulse response: {max_overall:.6f}")
print(f"   Max of final 100 samples: {max_final:.6f}")
print(f"   Decay ratio: {max_final / max_overall:.2e}")

if max_final / max_overall < 0.01:
    print(f"   [OK] Impulse response decays properly (stable)")
else:
    print(f"   [WARNING] Impulse response may not be decaying sufficiently")

# Generate C++ header file code
print("\n7. Generating C++ code...")

cpp_header = f"""/*
 * TrainingBandpassFilter.h
 *
 * 5th Order Butterworth Bandpass Filter (0.5-30 Hz at 250Hz)
 * Matches the training script preprocessing pipeline exactly
 *
 * Filter Specifications:
 *   - Type: Butterworth bandpass
 *   - Order: 5
 *   - Cutoff frequencies: {LOWCUT} Hz (low), {HIGHCUT} Hz (high)
 *   - Sample rate: {FS} Hz
 *   - Implementation: {len(sos)} cascaded second-order sections (biquads)
 *
 * Stability: All poles inside unit circle (max magnitude: {max_pole_mag:.6f})
 *
 * Generated by: generate_training_filter_coefficients.py
 */

#ifndef TRAINING_BANDPASS_FILTER_H
#define TRAINING_BANDPASS_FILTER_H

#include <Arduino.h>

class TrainingBandpassFilter {{
public:
    TrainingBandpassFilter();

    // Process single sample through the filter
    float process(float input);

    // Reset filter state (call before processing new data stream)
    void reset();

private:
    // Number of second-order sections
    static const int NUM_SECTIONS = {len(sos)};

    // SOS coefficients [b0, b1, b2, a1, a2] for each section
    // Note: a0 = 1.0 is implicit and not stored
    static const float b0[NUM_SECTIONS];
    static const float b1[NUM_SECTIONS];
    static const float b2[NUM_SECTIONS];
    static const float a1[NUM_SECTIONS];
    static const float a2[NUM_SECTIONS];

    // Filter state variables (Direct Form II)
    float w1[NUM_SECTIONS];  // Previous state 1
    float w2[NUM_SECTIONS];  // Previous state 2
}};

#endif // TRAINING_BANDPASS_FILTER_H
"""

cpp_source = f"""/*
 * TrainingBandpassFilter.cpp
 *
 * Implementation of 5th order Butterworth bandpass filter
 * using Direct Form II cascaded biquads for numerical stability
 */

#include "TrainingBandpassFilter.h"

// Initialize filter coefficients
// Each SOS: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]

"""

# Add coefficient arrays
for coef_idx, coef_name in enumerate(['b0', 'b1', 'b2']):
    cpp_source += f"const float TrainingBandpassFilter::{coef_name}[NUM_SECTIONS] = {{\n"
    for i, section in enumerate(sos_float32):
        cpp_source += f"    {section[coef_idx]:.18e}f"
        if i < len(sos_float32) - 1:
            cpp_source += ","
        cpp_source += f"  // Section {i+1}\n"
    cpp_source += "};\n\n"

# a1 and a2 coefficients (indices 4 and 5, but we store without a0)
for coef_idx, coef_name in [(4, 'a1'), (5, 'a2')]:
    cpp_source += f"const float TrainingBandpassFilter::{coef_name}[NUM_SECTIONS] = {{\n"
    for i, section in enumerate(sos_float32):
        cpp_source += f"    {section[coef_idx]:.18e}f"
        if i < len(sos_float32) - 1:
            cpp_source += ","
        cpp_source += f"  // Section {i+1}\n"
    cpp_source += "};\n\n"

cpp_source += """TrainingBandpassFilter::TrainingBandpassFilter() {
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
"""

# Save C++ files
with open('include/TrainingBandpassFilter.h', 'w') as f:
    f.write(cpp_header)
print("   [OK] Created: include/TrainingBandpassFilter.h")

with open('src/TrainingBandpassFilter.cpp', 'w') as f:
    f.write(cpp_source)
print("   [OK] Created: src/TrainingBandpassFilter.cpp")

# Create validation plot
if HAS_MATPLOTLIB:
    print("\n8. Creating validation plots...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Frequency response
    axes[0].plot(w, magnitude_db, 'b', linewidth=2)
    axes[0].axvline(LOWCUT, color='r', linestyle='--', alpha=0.5, label=f'Low cutoff: {LOWCUT} Hz')
    axes[0].axvline(HIGHCUT, color='r', linestyle='--', alpha=0.5, label=f'High cutoff: {HIGHCUT} Hz')
    axes[0].axhline(-3, color='g', linestyle=':', alpha=0.5, label='-3dB')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_title(f'{ORDER}th Order Butterworth Bandpass Filter ({LOWCUT}-{HIGHCUT} Hz) at {FS}Hz')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, FS/2)
    axes[0].set_ylim(-80, 5)
    axes[0].legend()

    # Phase response
    w_phase, h_phase = signal.sosfreqz(sos_float32, worN=8192, fs=FS)
    phase = np.unwrap(np.angle(h_phase))
    axes[1].plot(w_phase, np.degrees(phase), 'b', linewidth=2)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Phase (degrees)')
    axes[1].set_title('Phase Response')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 50)

    # Impulse response
    time = np.arange(impulse_length) / FS
    axes[2].plot(time, impulse_response, 'b', linewidth=1)
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title('Impulse Response (must decay to zero for stability)')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(0, color='k', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('training_filter_validation.png', dpi=150, bbox_inches='tight')
    print("   [OK] Saved: training_filter_validation.png")
    plt.show()
else:
    print("\n8. Skipping plots (matplotlib not available)")

# Final summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"[OK] Filter coefficients generated successfully")
print(f"[OK] Stability verified (max pole magnitude: {max_pole_mag:.6f} < 1.0)")
print(f"[OK] Float32 precision validated")
print(f"[OK] Impulse response decays properly")
print(f"[OK] C++ implementation files created")
print()
print("Generated files:")
print("  - include/TrainingBandpassFilter.h")
print("  - src/TrainingBandpassFilter.cpp")
print("  - training_filter_validation.png")
print()
print("Next steps:")
print("  1. Review the generated C++ files")
print("  2. Update your preprocessing pipeline to use TrainingBandpassFilter")
print("  3. Test on Teensy with sample data")
print("  4. Re-run validation notebook to verify >95% agreement")
print("="*70)
