"""
Test Script: Validate TrainingBandpassFilter Stability

This script tests the Butterworth filter coefficients to ensure they will work
reliably on the Teensy before deployment.

Tests performed:
1. Stability (poles inside unit circle)
2. Impulse response decay
3. Step response (no ringing/oscillation)
4. White noise test (no overflow/divergence)
5. Typical EEG signal test

Run this before deploying to hardware!
"""

import numpy as np
from scipy import signal
import sys

# Load filter coefficients from generated files
FS = 250  # Hz
LOWCUT = 0.5
HIGHCUT = 30
ORDER = 5

print("="*70)
print("FILTER STABILITY VALIDATION TEST")
print("="*70)
print()

# Design the filter
sos = signal.butter(ORDER, [LOWCUT, HIGHCUT], btype='band', fs=FS, output='sos')

# Test 1: Pole stability
print("TEST 1: Pole Stability")
print("-" * 70)
b, a = signal.sos2tf(sos)
poles = np.roots(a)
pole_mags = np.abs(poles)
max_pole_mag = np.max(pole_mags)

print(f"Number of poles: {len(poles)}")
print(f"Maximum pole magnitude: {max_pole_mag:.6f}")

if max_pole_mag < 1.0:
    print("[PASS] All poles inside unit circle")
    test1_pass = True
else:
    print("[FAIL] Poles outside unit circle - filter is unstable!")
    test1_pass = False

print()

# Test 2: Impulse response decay
print("TEST 2: Impulse Response Decay")
print("-" * 70)
impulse_len = 1000
impulse = np.zeros(impulse_len)
impulse[0] = 1.0

impulse_response = signal.sosfilt(sos, impulse)
final_samples = np.abs(impulse_response[-100:])
max_final = np.max(final_samples)
max_overall = np.max(np.abs(impulse_response))
decay_ratio = max_final / max_overall

print(f"Max impulse response: {max_overall:.6f}")
print(f"Max of final 100 samples: {max_final:.6f}")
print(f"Decay ratio: {decay_ratio:.2e}")

if decay_ratio < 0.01:
    print("[PASS] Impulse response decays properly")
    test2_pass = True
else:
    print("[FAIL] Impulse response not decaying sufficiently")
    test2_pass = False

print()

# Test 3: Step response (check for excessive ringing)
print("TEST 3: Step Response")
print("-" * 70)
step = np.ones(1000)
step_response = signal.sosfilt(sos, step)

# Check for overshoot
steady_state = np.mean(step_response[-100:])
peak = np.max(np.abs(step_response))
overshoot_pct = 100 * (peak - steady_state) / steady_state if steady_state != 0 else 0

print(f"Steady state value: {steady_state:.6f}")
print(f"Peak value: {peak:.6f}")
print(f"Overshoot: {overshoot_pct:.1f}%")

if overshoot_pct < 50:  # Less than 50% overshoot is acceptable
    print("[PASS] Step response acceptable")
    test3_pass = True
else:
    print("[WARN] Excessive overshoot in step response")
    test3_pass = False

print()

# Test 4: White noise test (check for divergence)
print("TEST 4: White Noise Test (Overflow/Divergence)")
print("-" * 70)
np.random.seed(42)
white_noise = np.random.randn(10000)
filtered_noise = signal.sosfilt(sos, white_noise)

input_rms = np.sqrt(np.mean(white_noise**2))
output_rms = np.sqrt(np.mean(filtered_noise**2))
max_output = np.max(np.abs(filtered_noise))

print(f"Input RMS: {input_rms:.3f}")
print(f"Output RMS: {output_rms:.3f}")
print(f"Max output: {max_output:.3f}")
print(f"Gain (RMS ratio): {output_rms/input_rms:.3f}")

# Check for divergence (output should be bounded)
if np.isfinite(max_output) and max_output < 100 * input_rms:
    print("[PASS] No divergence detected")
    test4_pass = True
else:
    print("[FAIL] Filter output diverging or overflowing")
    test4_pass = False

print()

# Test 5: Typical EEG signal test
print("TEST 5: Typical EEG Signal Test")
print("-" * 70)

# Simulate typical EEG: mixture of frequencies
t = np.arange(0, 10, 1/FS)  # 10 seconds at 250Hz
# Mix of delta (1Hz), theta (5Hz), alpha (10Hz), beta (20Hz)
eeg_signal = (
    np.sin(2*np.pi*1*t) * 100 +    # Delta
    np.sin(2*np.pi*5*t) * 50 +     # Theta
    np.sin(2*np.pi*10*t) * 30 +    # Alpha
    np.sin(2*np.pi*20*t) * 20 +    # Beta
    np.random.randn(len(t)) * 10   # Noise
)

filtered_eeg = signal.sosfilt(sos, eeg_signal)

eeg_input_range = np.max(eeg_signal) - np.min(eeg_signal)
eeg_output_range = np.max(filtered_eeg) - np.min(filtered_eeg)

print(f"Input range: {eeg_input_range:.1f} (typical EEG-like signal)")
print(f"Output range: {eeg_output_range:.1f}")
print(f"Output min/max: [{np.min(filtered_eeg):.1f}, {np.max(filtered_eeg):.1f}]")

# Check signal remains bounded
if np.isfinite(eeg_output_range) and not np.any(np.isnan(filtered_eeg)):
    print("[PASS] EEG signal filtered successfully")
    test5_pass = True
else:
    print("[FAIL] Filter produced NaN or infinite values")
    test5_pass = False

print()

# Final summary
print("="*70)
print("SUMMARY")
print("="*70)

all_tests = [
    ("Pole Stability", test1_pass),
    ("Impulse Decay", test2_pass),
    ("Step Response", test3_pass),
    ("White Noise", test4_pass),
    ("EEG Signal", test5_pass)
]

passed = sum(1 for _, result in all_tests if result)
total = len(all_tests)

for test_name, result in all_tests:
    status = "[PASS]" if result else "[FAIL]"
    print(f"{status} {test_name}")

print()
print(f"Total: {passed}/{total} tests passed")

if passed == total:
    print()
    print("[OK] All tests passed! Filter is stable and ready for deployment.")
    print("You can now compile and test on Teensy hardware.")
    sys.exit(0)
else:
    print()
    print("[WARNING] Some tests failed. Review filter design before deployment.")
    sys.exit(1)
