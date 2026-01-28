#!/usr/bin/env python3
"""
Compare filter coefficients and intermediate preprocessing values between
Python (scipy) and Teensy implementations to identify divergence sources.

This script performs three investigations:
1. Compare Butterworth filter coefficients
2. Compare intermediate preprocessing values for first epoch
3. Check if error accumulates over epochs
"""

import numpy as np
from scipy import signal
import struct
import os

# ============================================================================
# INVESTIGATION 1: Compare Filter Coefficients
# ============================================================================

def compare_filter_coefficients():
    """Compare Teensy hardcoded coefficients vs scipy-generated coefficients."""

    print("=" * 60)
    print("INVESTIGATION 1: Filter Coefficient Comparison")
    print("=" * 60)

    # Teensy coefficients (from TrainingBandpassFilter.cpp)
    # 5th order Butterworth bandpass, 0.5-30 Hz at 250 Hz
    teensy_b0 = np.array([
        2.579404041171073914e-03,
        1.000000000000000000e+00,
        1.000000000000000000e+00,
        1.000000000000000000e+00,
        1.000000000000000000e+00
    ])
    teensy_b1 = np.array([
        5.158808082342147827e-03,
        2.000000000000000000e+00,
        0.000000000000000000e+00,
        -2.000000000000000000e+00,
        -2.000000000000000000e+00
    ])
    teensy_b2 = np.array([
        2.579404041171073914e-03,
        1.000000000000000000e+00,
        -1.000000000000000000e+00,
        1.000000000000000000e+00,
        1.000000000000000000e+00
    ])
    teensy_a1 = np.array([
        -9.596837162971496582e-01,
        -1.212064862251281738e+00,
        -1.433070898056030273e+00,
        -1.979528427124023438e+00,
        -1.992304086685180664e+00
    ])
    teensy_a2 = np.array([
        2.994447648525238037e-01,
        6.596572399139404297e-01,
        4.402188658714294434e-01,
        9.796913862228393555e-01,
        9.924622774124145508e-01
    ])

    # Generate scipy coefficients
    fs = 250  # Sample rate
    low = 0.5
    high = 30
    order = 5

    sos_scipy = signal.butter(order, [low, high], btype='band', fs=fs, output='sos')

    print(f"\nFilter specs: {order}th order Butterworth bandpass, {low}-{high} Hz at {fs} Hz")
    print(f"Number of sections: Teensy={len(teensy_b0)}, scipy={len(sos_scipy)}")

    print("\n--- SOS Coefficient Comparison ---")
    print("Section | Coeff | Teensy          | scipy           | Diff            | Match?")
    print("-" * 85)

    max_diff = 0
    for i in range(min(len(teensy_b0), len(sos_scipy))):
        scipy_row = sos_scipy[i]
        # scipy SOS format: [b0, b1, b2, a0, a1, a2] where a0=1

        comparisons = [
            ('b0', teensy_b0[i], scipy_row[0]),
            ('b1', teensy_b1[i], scipy_row[1]),
            ('b2', teensy_b2[i], scipy_row[2]),
            ('a1', teensy_a1[i], scipy_row[4]),  # a1 is at index 4
            ('a2', teensy_a2[i], scipy_row[5]),  # a2 is at index 5
        ]

        for name, teensy_val, scipy_val in comparisons:
            diff = abs(teensy_val - scipy_val)
            max_diff = max(max_diff, diff)
            match = "YES" if diff < 1e-6 else ("~" if diff < 1e-3 else "NO")
            print(f"   {i+1}   |  {name}  | {teensy_val:15.10e} | {scipy_val:15.10e} | {diff:15.10e} | {match}")

    print(f"\nMaximum coefficient difference: {max_diff:.2e}")

    if max_diff < 1e-6:
        print("RESULT: Coefficients match exactly (within float64 precision)")
    elif max_diff < 1e-3:
        print("RESULT: Coefficients match closely (float32 precision loss)")
    else:
        print("RESULT: Coefficients DIFFER significantly!")

    return sos_scipy


# ============================================================================
# INVESTIGATION 2: Compare Intermediate Values for First Epoch
# ============================================================================

def load_raw_data(data_dir, max_samples=None):
    """Load the raw 4kHz data from SdioLogger file."""
    filepath = os.path.join(data_dir, "example_datasets", "eeg", "SdioLogger_miklos_night_2.bin")

    if not os.path.exists(filepath):
        print(f"ERROR: Raw data file not found: {filepath}")
        return None

    # Read only what we need (9 channels, int32 format = 36 bytes per sample)
    num_channels = 9
    bytes_per_sample = num_channels * 4

    if max_samples is None:
        # Only load first 5 epochs worth of data (5 * 30s * 4000Hz = 600,000 samples)
        max_samples = 600000

    bytes_to_read = max_samples * bytes_per_sample

    with open(filepath, 'rb') as f:
        raw_bytes = f.read(bytes_to_read)

    # Parse as int32 values
    num_values = len(raw_bytes) // 4
    all_data = np.array(struct.unpack(f'<{num_values}i', raw_bytes))

    # Reshape to (samples, channels)
    num_samples = num_values // num_channels
    data = all_data[:num_samples * num_channels].reshape(num_samples, num_channels)

    # Create bipolar derivation (CH0 - CH6)
    bipolar = data[:, 0] - data[:, 6]

    print(f"Loaded {len(bipolar)} samples ({len(bipolar)/4000:.1f} seconds at 4kHz)")
    return bipolar.astype(np.float32)


def teensy_style_preprocessing(raw_4khz, sos_coeffs, num_100hz_samples=3000):
    """
    Replicate Teensy preprocessing pipeline exactly:
    1. 4kHz -> 250Hz: take every 16th sample
    2. Filter at 250Hz using SOS coefficients
    3. 250Hz -> 100Hz: 5:2 resampling with interpolation
    """

    # Stage 1: Downsample 4kHz -> 250Hz (take every 16th sample)
    samples_250hz = raw_4khz[::16].copy()

    # Stage 2: Apply filter at 250Hz
    filtered_250hz = signal.sosfilt(sos_coeffs, samples_250hz)

    # Stage 3: Resample 250Hz -> 100Hz (5:2 ratio with interpolation)
    # Teensy takes sample 0 directly, then interpolates (sample[2] + sample[3]) / 2
    output_100hz = []
    i = 0
    while len(output_100hz) < num_100hz_samples and i + 4 < len(filtered_250hz):
        # First output: sample directly
        output_100hz.append(filtered_250hz[i])
        # Second output: interpolate between samples 2 and 3 (0-indexed from current position)
        output_100hz.append((filtered_250hz[i + 2] + filtered_250hz[i + 3]) / 2)
        i += 5

    return np.array(output_100hz[:num_100hz_samples]), samples_250hz, filtered_250hz


def compare_intermediate_values(data_dir):
    """Compare Python and Teensy intermediate values for epoch 0."""

    print("\n" + "=" * 60)
    print("INVESTIGATION 2: Intermediate Value Comparison (Epoch 0)")
    print("=" * 60)

    # Load raw data
    raw_4khz = load_raw_data(data_dir)
    if raw_4khz is None:
        return

    # Load reference preprocessed data (already at 100Hz, bandpassed)
    ref_path = os.path.join(data_dir, "example_datasets", "debug", "1_bandpassed_eeg_single_channel.npy")
    if os.path.exists(ref_path):
        ref_100hz = np.load(ref_path)
        print(f"Loaded reference 100Hz data: {len(ref_100hz)} samples")
    else:
        ref_100hz = None
        print(f"Reference 100Hz data not found")

    # Get scipy SOS coefficients (matching Teensy)
    sos_scipy = signal.butter(5, [0.5, 30], btype='band', fs=250, output='sos')

    # Process first 30 seconds (120,000 samples at 4kHz)
    epoch_4khz = raw_4khz[:120000]

    # Apply microvolt conversion (matching Teensy EEGFileReader.cpp)
    # voltage_uV = value * (2.0 * vref / gain) / (1 << 24) * 1e6
    gain = 24.0
    vref = 4.5
    scale_factor = (2.0 * vref / gain) / (1 << 24) * 1e6
    epoch_4khz_uv = epoch_4khz * scale_factor

    # Process using Teensy-style preprocessing (with microvolt scaling)
    output_100hz, samples_250hz, filtered_250hz = teensy_style_preprocessing(epoch_4khz_uv, sos_scipy)

    print(f"\nEpoch 0 processing (with microvolt conversion):")
    print(f"  Input: {len(epoch_4khz)} samples at 4kHz (30 seconds)")
    print(f"  Scale factor (ADC to uV): {scale_factor:.10e}")
    print(f"  After 250Hz downsample: {len(samples_250hz)} samples")
    print(f"  After filter: {len(filtered_250hz)} samples")
    print(f"  After 100Hz resample: {len(output_100hz)} samples")

    # Statistics matching Teensy checkpoint output
    print(f"\n--- 100Hz Preprocessed Signal (Python with uV conversion) ---")
    print(f"mean={np.mean(output_100hz):.4f} std={np.std(output_100hz):.4f} "
          f"min={np.min(output_100hz):.4f} max={np.max(output_100hz):.4f}")
    print(f"first_10: {' '.join([f'{x:.2f}' for x in output_100hz[:10]])}")
    print(f"last_10: {' '.join([f'{x:.2f}' for x in output_100hz[-10:]])}")

    # Compare with Teensy checkpoint values (from user's output)
    print(f"\n--- Teensy Checkpoint A (from validation run) ---")
    print(f"mean=0.3357 std=750.0453 min=-5216.0000 max=13165.1758")
    print(f"first_10: 36.23 2308.22 12628.62 13165.18 8834.16 8399.91 7074.69 5765.50 4594.13 3705.25")
    print(f"last_10: -21.28 -22.41 -30.67 -33.46 -29.72 -31.53 -34.77 -33.55 -33.70 -39.61")

    # Teensy values for comparison
    teensy_first_10 = np.array([36.23, 2308.22, 12628.62, 13165.18, 8834.16, 8399.91, 7074.69, 5765.50, 4594.13, 3705.25])
    teensy_last_10 = np.array([-21.28, -22.41, -30.67, -33.46, -29.72, -31.53, -34.77, -33.55, -33.70, -39.61])

    # Calculate differences
    print(f"\n--- Comparison (Python vs Teensy) ---")
    first_10_diff = output_100hz[:10] - teensy_first_10
    last_10_diff = output_100hz[-10:] - teensy_last_10

    print(f"First 10 difference:")
    print(f"  {' '.join([f'{x:.2f}' for x in first_10_diff])}")
    print(f"  Max abs diff: {np.max(np.abs(first_10_diff)):.4f}")

    print(f"\nLast 10 difference:")
    print(f"  {' '.join([f'{x:.2f}' for x in last_10_diff])}")
    print(f"  Max abs diff: {np.max(np.abs(last_10_diff)):.4f}")

    # Check correlation
    corr_first = np.corrcoef(output_100hz[:10], teensy_first_10)[0, 1]
    corr_last = np.corrcoef(output_100hz[-10:], teensy_last_10)[0, 1]
    print(f"\nCorrelation: first_10={corr_first:.6f}, last_10={corr_last:.6f}")

    # Compare with reference preprocessed data
    if ref_100hz is not None:
        print(f"\n--- Reference Data Comparison ---")
        ref_epoch0 = ref_100hz[:3000]
        print(f"Reference epoch 0 stats: mean={np.mean(ref_epoch0):.4f}, std={np.std(ref_epoch0):.4f}")
        print(f"Reference first_10: {' '.join([f'{x:.2f}' for x in ref_epoch0[:10]])}")
        print(f"Reference last_10: {' '.join([f'{x:.2f}' for x in ref_epoch0[-10:]])}")

        # Compare normalized versions (should match after Z-score)
        norm_python = (output_100hz - np.mean(output_100hz)) / np.std(output_100hz)
        norm_teensy = (teensy_first_10 - 0.3357) / 750.0453  # Using Teensy checkpoint stats
        norm_ref = (ref_epoch0[:10] - np.mean(ref_epoch0)) / np.std(ref_epoch0)

        print(f"\n--- Normalized Comparison (first 10) ---")
        print(f"Python (normalized): {' '.join([f'{x:.4f}' for x in norm_python[:10]])}")
        print(f"Reference (normalized): {' '.join([f'{x:.4f}' for x in norm_ref])}")
        print(f"Correlation Python vs Reference: {np.corrcoef(norm_python[:10], norm_ref)[0,1]:.6f}")

    return output_100hz


# ============================================================================
# INVESTIGATION 3: Error Accumulation Over Epochs
# ============================================================================

def check_error_accumulation(data_dir):
    """Check if prediction disagreement increases over epochs."""

    print("\n" + "=" * 60)
    print("INVESTIGATION 3: Error Accumulation Over Epochs")
    print("=" * 60)

    # Load reference predictions
    ref_path = os.path.join(data_dir, "example_datasets", "debug", "3_quantized_model_predictions.npy")
    if not os.path.exists(ref_path):
        print(f"ERROR: Reference predictions not found: {ref_path}")
        return

    ref_preds = np.load(ref_path)
    print(f"Loaded {len(ref_preds)} reference predictions")

    # From validation output - Teensy had 178 mismatches out of 959 epochs
    # Let's check if there's a pattern in the epoch numbers

    # Analyze by epoch ranges
    print("\nIf error accumulates, later epochs should have more mismatches.")
    print("From validation: 781/959 correct (81.4%)")
    print("\nExpected pattern if error accumulates:")
    print("  - Early epochs: high agreement")
    print("  - Later epochs: lower agreement")
    print("\nNote: First 10 epochs showed 100% agreement in validation.")
    print("This suggests error may accumulate over time.")

    # We can't recompute without running inference, but we can suggest
    # what to look for in the validation output
    print("\n--- Recommendation ---")
    print("Look at mismatch epochs from validation output.")
    print("If mismatches are clustered in later epochs, filter state accumulation is likely.")
    print("If mismatches are evenly distributed, precision differences are the cause.")


# ============================================================================
# INVESTIGATION 4: Float32 vs Float64 Precision
# ============================================================================

def test_precision_impact(data_dir):
    """Test if float32 precision causes significant drift."""

    print("\n" + "=" * 60)
    print("INVESTIGATION 4: Float32 vs Float64 Precision Impact")
    print("=" * 60)

    # Load raw data
    raw_4khz = load_raw_data(data_dir)
    if raw_4khz is None:
        return

    # Get scipy SOS coefficients in both precisions
    sos_f64 = signal.butter(5, [0.5, 30], btype='band', fs=250, output='sos')
    sos_f32 = sos_f64.astype(np.float32)

    # Process first epoch in both precisions
    epoch_4khz = raw_4khz[:120000]

    # Float64 processing
    samples_250hz = epoch_4khz[::16].astype(np.float64)
    filtered_f64 = signal.sosfilt(sos_f64, samples_250hz)

    # Float32 processing (matching Teensy)
    samples_250hz_f32 = epoch_4khz[::16].astype(np.float32)
    filtered_f32 = signal.sosfilt(sos_f32, samples_250hz_f32)

    # Compare
    diff = np.abs(filtered_f64 - filtered_f32.astype(np.float64))

    print(f"\nFilter output comparison (Epoch 0, {len(filtered_f64)} samples at 250Hz):")
    print(f"  Float64 mean: {np.mean(filtered_f64):.6f}")
    print(f"  Float32 mean: {np.mean(filtered_f32):.6f}")
    print(f"  Max absolute difference: {np.max(diff):.6e}")
    print(f"  Mean absolute difference: {np.mean(diff):.6e}")
    print(f"  RMS difference: {np.sqrt(np.mean(diff**2)):.6e}")

    # Check if difference grows over the epoch
    chunk_size = len(diff) // 10
    print(f"\n  Difference by position (to check accumulation):")
    for i in range(10):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk_diff = np.mean(diff[start:end])
        print(f"    Chunk {i+1} (samples {start}-{end}): mean diff = {chunk_diff:.6e}")

    # Check correlation
    corr = np.corrcoef(filtered_f64, filtered_f32)[0, 1]
    print(f"\n  Correlation between f64 and f32: {corr:.10f}")

    if np.max(diff) < 1e-3:
        print("\nRESULT: Float32 precision has minimal impact on filter output")
    else:
        print("\nRESULT: Float32 precision causes measurable differences")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Find data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, "data")

    print("=" * 60)
    print("Teensy vs Python Preprocessing Comparison")
    print("=" * 60)
    print(f"Data directory: {data_dir}")

    # Run investigations
    sos_scipy = compare_filter_coefficients()
    compare_intermediate_values(data_dir)
    check_error_accumulation(data_dir)
    test_precision_impact(data_dir)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Potential causes of 5% gap (86.8% Python vs 81.4% Teensy):

1. FILTER COEFFICIENTS: Check if they match exactly
   - If not, regenerate Teensy coefficients from scipy

2. INTERMEDIATE VALUES: Check first/last 10 samples match
   - Large differences indicate pipeline bug
   - Small differences indicate precision issues

3. ERROR ACCUMULATION: Check if later epochs have more errors
   - If yes, filter state precision is accumulating error
   - Consider periodic filter reset or double precision

4. FLOAT32 PRECISION: Check if f32 vs f64 causes drift
   - If significant, consider critical operations in double

Next steps based on findings:
- If coefficients differ: Update Teensy coefficients
- If accumulation detected: Reset filter between epochs
- If precision drift: Use double for filter state variables
""")
