#!/usr/bin/env python3
"""
Phase 1: Python Prediction Agreement Test
==========================================

This script tests whether our Python preprocessing of the raw SdioLogger data
produces predictions that match the reference predictions.

The critical insight is that even if sample-level values differ slightly,
the ML model may still produce the same predictions after Z-score normalization.

Usage:
    python tools/test_prediction_agreement.py

Expected output:
    - Agreement >= 95%: Our preprocessing is "close enough" for the model
    - Agreement < 95%: Need to investigate preprocessing differences further
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter, resample
from pathlib import Path
import sys

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Try to import TensorFlow
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    print("WARNING: TensorFlow not installed. Install with: pip install tensorflow")
    HAS_TF = False

# Configuration matching firmware and training script
DATA_FILE_4KHZ = Path('data/example_datasets/eeg/SdioLogger_miklos_night_2.bin')
DATA_FILE_100HZ = Path('data/example_datasets/eeg/SdioLogger_miklos_night_2_Fs_100Hz.bin')
REFERENCE_EEG = Path('data/example_datasets/debug/1_bandpassed_eeg_single_channel.npy')
REFERENCE_NORMALIZED = Path('data/example_datasets/debug/2_standardized_epochs.npy')
REFERENCE_PREDICTIONS = Path('data/example_datasets/debug/3_quantized_model_predictions.npy')
REFERENCE_PROBABILITIES = Path('data/example_datasets/debug/4_quantized_model_probabilities.npy')
MODEL_FILE = Path('data/example_datasets/debug/8_tflite_quantized_model.tflite')

# Signal parameters
SAMPLE_RATE_ORIGINAL = 4000  # Hz (raw file)
SAMPLE_RATE_FILTER = 250     # Hz (filter designed for this rate)
SAMPLE_RATE_TARGET = 100     # Hz (model input)
NUM_CHANNELS = 9
BIPOLAR_POS = 0  # Ch0 (0-indexed)
BIPOLAR_NEG = 6  # Ch6 (0-indexed)
DATA_DTYPE = np.int32

# Filter parameters (from training script)
LOWCUT = 0.5   # Hz
HIGHCUT = 30   # Hz
FILTER_ORDER = 5

# Epoch parameters
EPOCH_LENGTH_SEC = 30
EPOCH_SAMPLES_100HZ = EPOCH_LENGTH_SEC * SAMPLE_RATE_TARGET  # 3000 samples

# Sleep stage names
STAGE_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}")


def butter_bandpass(lowcut, highcut, fs, order=5):
    """Create bandpass filter coefficients (matching training script exactly)"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply bandpass filter (matching training script exactly)"""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def load_raw_4khz_data():
    """Load raw 4kHz data from binary file"""
    print_section("Loading Raw 4kHz Data")

    if not DATA_FILE_4KHZ.exists():
        print(f"ERROR: Raw data file not found: {DATA_FILE_4KHZ}")
        return None

    raw_data = np.fromfile(DATA_FILE_4KHZ, dtype=DATA_DTYPE)
    num_samples = len(raw_data) // NUM_CHANNELS
    data_multi_channel = raw_data[:num_samples * NUM_CHANNELS].reshape(-1, NUM_CHANNELS)

    # Extract bipolar channel (Ch0 - Ch6)
    bipolar_data = data_multi_channel[:, BIPOLAR_POS] - data_multi_channel[:, BIPOLAR_NEG]

    print(f"Loaded file: {DATA_FILE_4KHZ}")
    print(f"Total samples per channel: {num_samples:,}")
    print(f"Duration: {num_samples / SAMPLE_RATE_ORIGINAL / 3600:.2f} hours")
    print(f"Bipolar data shape: {bipolar_data.shape}")
    print(f"Bipolar data range: [{bipolar_data.min()}, {bipolar_data.max()}]")

    return bipolar_data.astype(np.float64)


def preprocess_method_training_script(bipolar_4khz):
    """
    Preprocess using the EXACT method from the training script:
    1. Downsample 4kHz -> 250Hz (average every 16 samples)
    2. Apply bandpass filter at 250Hz
    3. Resample 250Hz -> 100Hz using scipy.signal.resample

    This matches lines 156-157 of 5_training_script.py:
        x = butter_bandpass_filter(x, freq_low, freq_high, 250)
        x_r = resample(x, int(len(x)*fs_d/250))
    """
    print_section("Preprocessing (Training Script Method)")

    # Step 1: Downsample 4kHz -> 250Hz by averaging
    num_samples_250hz = len(bipolar_4khz) // 16
    data_250hz = np.zeros(num_samples_250hz, dtype=np.float64)
    for i in range(num_samples_250hz):
        start = i * 16
        data_250hz[i] = np.mean(bipolar_4khz[start:start+16])

    print(f"After 4kHz->250Hz downsample: {len(data_250hz):,} samples")

    # Step 2: Apply bandpass filter at 250Hz (matching training script)
    filtered_250hz = butter_bandpass_filter(data_250hz, LOWCUT, HIGHCUT, 250, order=FILTER_ORDER)

    print(f"After bandpass filter (0.5-30Hz): {len(filtered_250hz):,} samples")

    # Step 3: Resample 250Hz -> 100Hz using scipy.signal.resample (matching training script)
    num_samples_100hz = int(len(filtered_250hz) * SAMPLE_RATE_TARGET / SAMPLE_RATE_FILTER)
    data_100hz = resample(filtered_250hz, num_samples_100hz)

    print(f"After 250Hz->100Hz resample: {len(data_100hz):,} samples")
    print(f"Duration at 100Hz: {len(data_100hz) / SAMPLE_RATE_TARGET / 3600:.2f} hours")

    return data_100hz.astype(np.float32)


def extract_and_normalize_epochs(data_100hz, num_epochs=None):
    """Extract 30-second epochs and apply Z-score normalization (per-epoch)"""

    max_epochs = len(data_100hz) // EPOCH_SAMPLES_100HZ
    if num_epochs is None or num_epochs > max_epochs:
        num_epochs = max_epochs

    print(f"\nExtracting {num_epochs} epochs...")

    epochs_normalized = np.zeros((num_epochs, EPOCH_SAMPLES_100HZ), dtype=np.float32)
    epoch_means = np.zeros(num_epochs)
    epoch_stds = np.zeros(num_epochs)

    for i in range(num_epochs):
        start = i * EPOCH_SAMPLES_100HZ
        end = start + EPOCH_SAMPLES_100HZ
        epoch = data_100hz[start:end]

        # Z-score normalization (matching training script lines 173-174)
        mean = np.mean(epoch)
        std = np.std(epoch)
        epochs_normalized[i] = (epoch - mean) / std
        epoch_means[i] = mean
        epoch_stds[i] = std

    print(f"Epochs shape: {epochs_normalized.shape}")
    print(f"Mean of normalized epochs (should be ~0): {np.mean(epochs_normalized):.6f}")
    print(f"Std of normalized epochs (should be ~1): {np.std(epochs_normalized):.6f}")

    return epochs_normalized, epoch_means, epoch_stds


def run_tflite_inference(epochs_normalized):
    """Run inference using TFLite model"""

    if not HAS_TF:
        print("ERROR: TensorFlow not available, cannot run inference")
        return None

    if not MODEL_FILE.exists():
        print(f"ERROR: Model file not found: {MODEL_FILE}")
        return None

    print_section("Running TFLite Inference")

    # Load model
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_FILE))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Model loaded: {MODEL_FILE}")
    print(f"Input 0 (EEG): shape={input_details[0]['shape']}, dtype={input_details[0]['dtype']}")
    print(f"Input 1 (Epoch): shape={input_details[1]['shape']}, dtype={input_details[1]['dtype']}")
    print(f"Output: shape={output_details[0]['shape']}, dtype={output_details[0]['dtype']}")

    num_epochs = len(epochs_normalized)
    predictions = np.zeros(num_epochs, dtype=np.int32)
    probabilities = np.zeros((num_epochs, 5), dtype=np.float32)

    print(f"\nProcessing {num_epochs} epochs...")

    for i in range(num_epochs):
        # Prepare EEG input (shape: 1, 1, 3000, 1)
        eeg_input = epochs_normalized[i].reshape(1, 1, 3000, 1).astype(np.float32)

        # Prepare epoch index input (scaled by /1000 as in training script pos_var_v2)
        epoch_input = np.array([[i / 1000.0]], dtype=np.float32)

        # Set inputs
        interpreter.set_tensor(input_details[0]['index'], eeg_input)
        interpreter.set_tensor(input_details[1]['index'], epoch_input)

        # Run inference
        interpreter.invoke()

        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        probabilities[i] = output
        predictions[i] = np.argmax(output)

        # Progress update
        if (i + 1) % 100 == 0 or i == num_epochs - 1:
            print(f"  Processed {i + 1}/{num_epochs} epochs ({100 * (i + 1) / num_epochs:.1f}%)")

    return predictions, probabilities


def compare_predictions(our_predictions, ref_predictions, our_probabilities=None, ref_probabilities=None):
    """Compare our predictions with reference predictions"""

    print_section("Prediction Comparison Results")

    num_epochs = min(len(our_predictions), len(ref_predictions))

    # Calculate agreement
    matches = np.sum(our_predictions[:num_epochs] == ref_predictions[:num_epochs])
    agreement = matches / num_epochs * 100

    print(f"Total epochs compared: {num_epochs}")
    print(f"Matching predictions: {matches}")
    print(f"Agreement: {agreement:.2f}%")

    # Find mismatches
    mismatches = np.where(our_predictions[:num_epochs] != ref_predictions[:num_epochs])[0]

    if len(mismatches) > 0:
        print(f"\nMismatched epochs: {len(mismatches)}")
        print("\nFirst 20 mismatches:")
        print(f"{'Epoch':<8} {'Our':<8} {'Ref':<8} {'Our Prob':<30} {'Ref Prob':<30}")
        print("-" * 90)

        for i, epoch in enumerate(mismatches[:20]):
            our_stage = STAGE_NAMES[our_predictions[epoch]]
            ref_stage = STAGE_NAMES[ref_predictions[epoch]]

            if our_probabilities is not None and ref_probabilities is not None:
                our_prob_str = " ".join([f"{p:.2f}" for p in our_probabilities[epoch]])
                ref_prob_str = " ".join([f"{p:.2f}" for p in ref_probabilities[epoch]])
            else:
                our_prob_str = "N/A"
                ref_prob_str = "N/A"

            print(f"{epoch:<8} {our_stage:<8} {ref_stage:<8} {our_prob_str:<30} {ref_prob_str:<30}")
    else:
        print("\n*** PERFECT AGREEMENT - All predictions match! ***")

    # Per-class analysis
    print("\nPer-class agreement:")
    for stage_idx, stage_name in enumerate(STAGE_NAMES):
        ref_mask = ref_predictions[:num_epochs] == stage_idx
        if np.sum(ref_mask) > 0:
            class_matches = np.sum(our_predictions[:num_epochs][ref_mask] == stage_idx)
            class_total = np.sum(ref_mask)
            class_agreement = class_matches / class_total * 100
            print(f"  {stage_name}: {class_matches}/{class_total} ({class_agreement:.1f}%)")

    return agreement, mismatches


def compare_100hz_signal(our_100hz, ref_100hz):
    """Compare our 100Hz signal with reference"""

    print_section("100Hz Signal Comparison")

    num_samples = min(len(our_100hz), len(ref_100hz))

    # Calculate correlation
    correlation = np.corrcoef(our_100hz[:num_samples], ref_100hz[:num_samples])[0, 1]

    # Calculate per-epoch correlation
    num_epochs = num_samples // EPOCH_SAMPLES_100HZ
    epoch_correlations = []
    epoch_scale_ratios = []

    for i in range(min(num_epochs, 10)):  # First 10 epochs
        start = i * EPOCH_SAMPLES_100HZ
        end = start + EPOCH_SAMPLES_100HZ

        our_epoch = our_100hz[start:end]
        ref_epoch = ref_100hz[start:end]

        corr = np.corrcoef(our_epoch, ref_epoch)[0, 1]
        scale_ratio = np.std(our_epoch) / np.std(ref_epoch)

        epoch_correlations.append(corr)
        epoch_scale_ratios.append(scale_ratio)

    print(f"Overall correlation: {correlation:.4f}")
    print(f"\nPer-epoch analysis (first 10 epochs):")
    print(f"{'Epoch':<8} {'Correlation':<15} {'Scale Ratio':<15}")
    print("-" * 40)

    for i in range(len(epoch_correlations)):
        print(f"{i:<8} {epoch_correlations[i]:<15.4f} {epoch_scale_ratios[i]:<15.4f}")

    # Check for DC offset differences
    our_mean = np.mean(our_100hz[:num_samples])
    ref_mean = np.mean(ref_100hz[:num_samples])
    print(f"\nDC offset comparison:")
    print(f"  Our mean: {our_mean:.4f}")
    print(f"  Ref mean: {ref_mean:.4f}")
    print(f"  Difference: {our_mean - ref_mean:.4f}")

    return correlation, epoch_correlations


def compare_normalized_epochs(our_normalized, ref_normalized):
    """Compare normalized epochs"""

    print_section("Normalized Epoch Comparison")

    num_epochs = min(len(our_normalized), len(ref_normalized))

    max_diffs = []
    mean_diffs = []
    correlations = []

    for i in range(num_epochs):
        diff = np.abs(our_normalized[i] - ref_normalized[i])
        max_diffs.append(np.max(diff))
        mean_diffs.append(np.mean(diff))
        correlations.append(np.corrcoef(our_normalized[i], ref_normalized[i])[0, 1])

    print(f"Epochs compared: {num_epochs}")
    print(f"\nDifference statistics:")
    print(f"  Max difference (overall): {max(max_diffs):.6f}")
    print(f"  Mean difference (overall): {np.mean(mean_diffs):.6f}")
    print(f"  Mean correlation: {np.mean(correlations):.6f}")

    # Show first 5 epochs in detail
    print(f"\nFirst 5 epochs detail:")
    for i in range(min(5, num_epochs)):
        print(f"  Epoch {i}: max_diff={max_diffs[i]:.6f}, mean_diff={mean_diffs[i]:.6f}, corr={correlations[i]:.6f}")

    return max_diffs, mean_diffs, correlations


def main():
    print("#" * 70)
    print("# Phase 1: Python Prediction Agreement Test")
    print("# Testing whether our preprocessing produces same predictions as reference")
    print("#" * 70)

    # Load reference data
    print_section("Loading Reference Data")

    ref_eeg = np.load(REFERENCE_EEG) if REFERENCE_EEG.exists() else None
    ref_normalized = np.load(REFERENCE_NORMALIZED) if REFERENCE_NORMALIZED.exists() else None
    ref_predictions = np.load(REFERENCE_PREDICTIONS) if REFERENCE_PREDICTIONS.exists() else None
    ref_probabilities = np.load(REFERENCE_PROBABILITIES) if REFERENCE_PROBABILITIES.exists() else None

    if ref_predictions is None:
        print("ERROR: Reference predictions not found!")
        return

    print(f"Reference EEG shape: {ref_eeg.shape if ref_eeg is not None else 'N/A'}")
    print(f"Reference normalized shape: {ref_normalized.shape if ref_normalized is not None else 'N/A'}")
    print(f"Reference predictions shape: {ref_predictions.shape}")
    print(f"Reference probabilities shape: {ref_probabilities.shape if ref_probabilities is not None else 'N/A'}")

    num_ref_epochs = len(ref_predictions)
    print(f"Number of reference epochs: {num_ref_epochs}")

    # APPROACH 1: Use reference preprocessed EEG directly
    # This tests if our inference matches when using the SAME preprocessed data
    print_section("APPROACH 1: Using Reference Preprocessed EEG")

    if ref_eeg is not None:
        # Extract and normalize epochs from reference EEG
        epochs_normalized, _, _ = extract_and_normalize_epochs(ref_eeg, num_ref_epochs)

        # Compare with reference normalized epochs
        if ref_normalized is not None:
            compare_normalized_epochs(epochs_normalized, ref_normalized)

        # Run inference
        if HAS_TF:
            our_predictions, our_probabilities = run_tflite_inference(epochs_normalized)

            if our_predictions is not None:
                agreement, mismatches = compare_predictions(
                    our_predictions, ref_predictions,
                    our_probabilities, ref_probabilities
                )

                if agreement >= 99.0:
                    print("\n" + "=" * 70)
                    print("SUCCESS: Using reference EEG achieves >= 99% agreement!")
                    print("This confirms our normalization and inference code is correct.")
                    print("=" * 70)

    # APPROACH 2: Process raw 4kHz data with our preprocessing
    # This tests if our complete preprocessing pipeline matches
    print_section("APPROACH 2: Processing Raw 4kHz Data")

    bipolar_4khz = load_raw_4khz_data()

    if bipolar_4khz is not None:
        # Preprocess using training script method
        our_100hz = preprocess_method_training_script(bipolar_4khz)

        # Compare 100Hz signal with reference
        if ref_eeg is not None:
            compare_100hz_signal(our_100hz, ref_eeg)

        # Extract and normalize epochs
        our_epochs_normalized, epoch_means, epoch_stds = extract_and_normalize_epochs(our_100hz, num_ref_epochs)

        # Compare with reference normalized epochs
        if ref_normalized is not None:
            compare_normalized_epochs(our_epochs_normalized, ref_normalized)

        # Run inference on our preprocessed data
        if HAS_TF:
            our_predictions, our_probabilities = run_tflite_inference(our_epochs_normalized)

            if our_predictions is not None:
                agreement, mismatches = compare_predictions(
                    our_predictions, ref_predictions,
                    our_probabilities, ref_probabilities
                )

                print("\n" + "=" * 70)
                if agreement >= 95.0:
                    print(f"SUCCESS: Our preprocessing achieves {agreement:.1f}% agreement!")
                    print("Proceed to Phase 2: Teensy validation")
                else:
                    print(f"NEEDS INVESTIGATION: Only {agreement:.1f}% agreement")
                    print("Our preprocessing differs significantly from reference")
                print("=" * 70)

    # Final summary
    print_section("SUMMARY")
    print("This test determines if our Python preprocessing matches the reference.")
    print("If Approach 1 shows high agreement but Approach 2 doesn't, the issue")
    print("is in our preprocessing pipeline (filtering, resampling).")
    print("If both show high agreement, proceed to Phase 2 (Teensy validation).")


if __name__ == '__main__':
    main()
