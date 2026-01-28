#!/usr/bin/env python3
"""
Compare preprocessing statistics between Python simulation and Teensy firmware.

This script extracts detailed statistics for the first 5 epochs from the
Python simulation to compare against the Teensy debug output.

Usage:
    python tools/compare_preprocessing.py

Expected output: Reference values to compare with Teensy serial output.
"""

import numpy as np
from scipy import signal
from pathlib import Path
import tensorflow as tf

# Configuration
DATA_FILE = Path('data/example_datasets/eeg/SdioLogger_miklos_night_2.bin')
MODEL_FILE = Path('data/example_datasets/debug/8_tflite_quantized_model.tflite')

SAMPLE_RATE_ORIGINAL = 4000  # Hz
SAMPLE_RATE_TARGET = 100     # Hz
NUM_CHANNELS = 9
BIPOLAR_POS = 0  # Ch1
BIPOLAR_NEG = 6  # Ch7
DATA_DTYPE = np.int32

# ADC to microvolt conversion (matching Teensy firmware EEGFileReader.cpp)
# Formula: value * (2 * vref / gain) / (2^24) * 1e6
ADC_GAIN = 24.0
ADC_VREF = 4.5
ADC_TO_UV_SCALE = (2.0 * ADC_VREF / ADC_GAIN) / (2**24) * 1e6  # ~0.02235

EPOCH_LENGTH_SEC = 30
EPOCH_LENGTH_SAMPLES = EPOCH_LENGTH_SEC * SAMPLE_RATE_TARGET  # 3000

TRAINING_FS = 250  # Hz
LOWCUT = 0.5       # Hz
HIGHCUT = 30       # Hz
FILTER_ORDER = 5

STAGE_LABELS = ['Wake', 'N1', 'N2', 'N3', 'REM']


class TrainingBandpassFilter:
    """Python implementation of TrainingBandpassFilter matching firmware exactly"""
    def __init__(self):
        # Filter coefficients from TrainingBandpassFilter.cpp
        self.b0 = np.array([
            2.579404041171073914e-03,
            1.000000000000000000e+00,
            1.000000000000000000e+00,
            1.000000000000000000e+00,
            1.000000000000000000e+00
        ], dtype=np.float32)

        self.b1 = np.array([
            5.158808082342147827e-03,
            2.000000000000000000e+00,
            0.000000000000000000e+00,
            -2.000000000000000000e+00,
            -2.000000000000000000e+00
        ], dtype=np.float32)

        self.b2 = np.array([
            2.579404041171073914e-03,
            1.000000000000000000e+00,
            -1.000000000000000000e+00,
            1.000000000000000000e+00,
            1.000000000000000000e+00
        ], dtype=np.float32)

        self.a1 = np.array([
            -9.596837162971496582e-01,
            -1.212064862251281738e+00,
            -1.433070898056030273e+00,
            -1.979528427124023438e+00,
            -1.992304086685180664e+00
        ], dtype=np.float32)

        self.a2 = np.array([
            2.994447648525238037e-01,
            6.596572399139404297e-01,
            4.402188658714294434e-01,
            9.796913862228393555e-01,
            9.924622774124145508e-01
        ], dtype=np.float32)

        self.w1 = np.zeros(5, dtype=np.float32)
        self.w2 = np.zeros(5, dtype=np.float32)

    def reset(self):
        self.w1 = np.zeros(5, dtype=np.float32)
        self.w2 = np.zeros(5, dtype=np.float32)

    def process(self, input_val):
        """Process single sample through cascaded biquads (Direct Form II)"""
        output = np.float32(input_val)

        for i in range(5):
            w0 = output - self.a1[i] * self.w1[i] - self.a2[i] * self.w2[i]
            output = self.b0[i] * w0 + self.b1[i] * self.w1[i] + self.b2[i] * self.w2[i]
            self.w2[i] = self.w1[i]
            self.w1[i] = w0

        return output


def load_and_preprocess_firmware_style(num_epochs_to_process=5):
    """
    Load data and preprocess using firmware-style pipeline.
    Returns preprocessed epochs for comparison.
    """
    print("=" * 70)
    print("FIRMWARE-STYLE PREPROCESSING (Python Simulation)")
    print("=" * 70)

    # Load binary file
    print(f"\nLoading: {DATA_FILE}")
    raw_data = np.fromfile(DATA_FILE, dtype=DATA_DTYPE)
    num_samples = len(raw_data) // NUM_CHANNELS
    data_multi_channel = raw_data[:num_samples * NUM_CHANNELS].reshape(-1, NUM_CHANNELS)

    # Extract bipolar channel (Ch1 - Ch7) and convert to microvolts
    # This matches Teensy firmware EEGFileReader.cpp line 316
    eeg_data_raw = data_multi_channel[:, BIPOLAR_POS] - data_multi_channel[:, BIPOLAR_NEG]
    eeg_data = eeg_data_raw.astype(np.float32) * ADC_TO_UV_SCALE
    print(f"Bipolar channel shape: {eeg_data.shape}")
    print(f"ADC to uV scale factor: {ADC_TO_UV_SCALE:.6f}")
    print(f"First raw bipolar: {eeg_data_raw[0]}, First uV bipolar: {eeg_data[0]:.4f}")

    # Step 1: Downsample 4kHz -> 250Hz (average every 16 samples)
    print("\nStep 1: Downsampling 4kHz -> 250Hz...")
    num_samples_250hz = len(eeg_data) // 16
    eeg_250hz = np.zeros(num_samples_250hz, dtype=np.float32)
    for i in range(num_samples_250hz):
        start_idx = i * 16
        eeg_250hz[i] = np.mean(eeg_data[start_idx:start_idx + 16])

    # Step 2: Apply TrainingBandpassFilter at 250Hz
    print("Step 2: Applying TrainingBandpassFilter...")
    training_filter = TrainingBandpassFilter()
    eeg_250hz_filtered = np.zeros_like(eeg_250hz)

    # Only process enough for the epochs we need
    samples_needed = (num_epochs_to_process * EPOCH_LENGTH_SAMPLES * TRAINING_FS // SAMPLE_RATE_TARGET) + 10000
    for i in range(min(len(eeg_250hz), samples_needed)):
        eeg_250hz_filtered[i] = training_filter.process(eeg_250hz[i])
        if (i + 1) % 100000 == 0:
            print(f"  Processed {i+1:,}/{min(len(eeg_250hz), samples_needed):,} samples...")

    # Step 3: Resample 250Hz -> 100Hz (5:2 ratio with linear interpolation)
    print("Step 3: Resampling 250Hz -> 100Hz...")
    num_groups = min(len(eeg_250hz_filtered) // 5, samples_needed // 5)
    eeg_100hz = np.zeros(num_groups * 2, dtype=np.float32)

    for i in range(num_groups):
        start_idx = i * 5
        group = eeg_250hz_filtered[start_idx:start_idx + 5]
        eeg_100hz[i * 2] = group[0]
        eeg_100hz[i * 2 + 1] = (group[2] + group[3]) * 0.5

    print(f"100Hz samples: {len(eeg_100hz):,}")

    # Create epochs
    epochs_100hz = []
    epochs_normalized = []

    for epoch_idx in range(num_epochs_to_process):
        start = epoch_idx * EPOCH_LENGTH_SAMPLES
        end = start + EPOCH_LENGTH_SAMPLES

        if end > len(eeg_100hz):
            print(f"Warning: Not enough samples for epoch {epoch_idx}")
            break

        epoch_data = eeg_100hz[start:end].copy()
        epochs_100hz.append(epoch_data)

        # Normalize
        mean = np.mean(epoch_data)
        std = np.std(epoch_data)
        epoch_norm = (epoch_data - mean) / std
        epochs_normalized.append(epoch_norm.astype(np.float32))

    return epochs_100hz, epochs_normalized


def run_model_inference(epochs_normalized):
    """Run TFLite model inference on normalized epochs."""
    print("\n" + "=" * 70)
    print("MODEL INFERENCE")
    print("=" * 70)

    # Load model
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_FILE))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get quantization parameters
    # Try multiple formats since TFLite versions differ
    input_scale = 0.0
    input_zero_point = 0

    quant_params = input_details[0].get('quantization_parameters', {})
    scales = quant_params.get('scales', [])
    zero_points = quant_params.get('zero_points', [])

    if len(scales) > 0:
        input_scale = scales[0]
    if len(zero_points) > 0:
        input_zero_point = zero_points[0]

    # Fallback to legacy format
    if input_scale == 0.0:
        legacy_quant = input_details[0].get('quantization', (0.0, 0))
        if len(legacy_quant) >= 2:
            input_scale = legacy_quant[0]
            input_zero_point = legacy_quant[1]

    print(f"\nInput quantization:")
    print(f"  Scale: {input_scale}")
    print(f"  Zero point: {input_zero_point}")

    # If still no quantization, use typical values from the Teensy firmware
    if input_scale == 0.0:
        print("  Note: Using typical INT8 quantization values from firmware")
        input_scale = 0.05  # Typical value; will need to verify from Teensy output

    results = []
    for epoch_idx, epoch_norm in enumerate(epochs_normalized):
        # Reshape for model
        eeg_input = epoch_norm.reshape(1, 1, 3000, 1).astype(np.float32)
        epoch_input = np.array([[epoch_idx]], dtype=np.float32)

        interpreter.set_tensor(input_details[0]['index'], eeg_input)
        interpreter.set_tensor(input_details[1]['index'], epoch_input)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])[0]
        results.append(output)

    return results, input_scale, input_zero_point


def print_epoch_debug(epoch_idx, epoch_100hz, epoch_normalized, model_output, scale, zero_point):
    """Print debug information for a single epoch (matching Teensy format)."""
    print("\n" + "=" * 60)
    print(f"=== PYTHON REFERENCE: Epoch {epoch_idx} ===")
    print("=" * 60)

    # 100Hz statistics (before normalization)
    print("\n--- Pre-Normalization Statistics (100Hz, 3000 samples) ---")
    print(f"  Mean:     {np.mean(epoch_100hz):.8f}")
    print(f"  Std Dev:  {np.std(epoch_100hz):.8f}")
    print(f"  Min:      {np.min(epoch_100hz):.8f}")
    print(f"  Max:      {np.max(epoch_100hz):.8f}")

    # Normalized window statistics
    print("\n--- Normalized Window Statistics (3000 samples) ---")
    print(f"  Mean:     {np.mean(epoch_normalized):.8f}")
    print(f"  Std Dev:  {np.std(epoch_normalized):.8f}")
    print(f"  Min:      {np.min(epoch_normalized):.8f}")
    print(f"  Max:      {np.max(epoch_normalized):.8f}")

    print("\n--- First 10 Normalized Samples ---")
    for i in range(10):
        print(f"  [{i}]: {epoch_normalized[i]:.8f}")

    print("\n--- Last 10 Normalized Samples ---")
    for i in range(2990, 3000):
        print(f"  [{i}]: {epoch_normalized[i]:.8f}")

    # Quantization
    if scale > 0:
        print("\n--- Quantization Parameters ---")
        print(f"  Scale:      {scale:.10f}")
        print(f"  Zero Point: {zero_point}")

        print("\n--- First 10 Quantized Samples (INT8) ---")
        for i in range(10):
            quantized = int(round(epoch_normalized[i] / scale)) + zero_point
            quantized = max(-128, min(127, quantized))
            print(f"  [{i}]: {epoch_normalized[i]:.6f} -> {quantized}")

        # Epoch index quantization
        epoch_scaled = epoch_idx / 1000.0
        epoch_quantized = int(round(epoch_scaled / scale)) + zero_point
        epoch_quantized = max(-128, min(127, epoch_quantized))
        print("\n--- Epoch Index Quantization ---")
        print(f"  Epoch index: {epoch_idx}")
        print(f"  Scaled (/1000): {epoch_scaled:.6f}")
        print(f"  Quantized: {epoch_quantized}")

    # Model output
    print("\n--- Model Output Probabilities ---")
    print(f"  Wake (yy0): {model_output[0]:.6f}")
    print(f"  N1   (yy1): {model_output[1]:.6f}")
    print(f"  N2   (yy2): {model_output[2]:.6f}")
    print(f"  N3   (yy3): {model_output[3]:.6f}")
    print(f"  REM  (yy4): {model_output[4]:.6f}")

    pred_idx = np.argmax(model_output)
    print(f"\n  Predicted: {STAGE_LABELS[pred_idx]} ({model_output[pred_idx]*100:.1f}%)")

    print("=" * 60)


def main():
    print("\n" + "#" * 70)
    print("#  PREPROCESSING COMPARISON REFERENCE VALUES")
    print("#  Compare these with Teensy debug output")
    print("#" * 70 + "\n")

    # Process first 5 epochs
    epochs_100hz, epochs_normalized = load_and_preprocess_firmware_style(num_epochs_to_process=5)

    # Run inference
    model_outputs, scale, zero_point = run_model_inference(epochs_normalized)

    # Print detailed debug for each epoch
    print("\n" + "#" * 70)
    print("#  REFERENCE VALUES FOR EPOCHS 0-4")
    print("#  Compare with Teensy serial output")
    print("#" * 70)

    for epoch_idx in range(len(epochs_normalized)):
        print_epoch_debug(
            epoch_idx,
            epochs_100hz[epoch_idx],
            epochs_normalized[epoch_idx],
            model_outputs[epoch_idx],
            scale,
            zero_point
        )

    # Summary comparison table
    print("\n\n" + "=" * 70)
    print("QUICK COMPARISON TABLE")
    print("=" * 70)
    print(f"\n{'Epoch':<8} {'Norm Mean':<12} {'Norm Std':<12} {'Predicted':<12} {'Confidence':<12}")
    print("-" * 70)

    for epoch_idx in range(len(epochs_normalized)):
        norm = epochs_normalized[epoch_idx]
        out = model_outputs[epoch_idx]
        pred_idx = np.argmax(out)
        print(f"{epoch_idx:<8} {np.mean(norm):<12.6f} {np.std(norm):<12.6f} "
              f"{STAGE_LABELS[pred_idx]:<12} {out[pred_idx]*100:<12.1f}%")

    print("=" * 70)
    print("\nUSAGE: Compare these values with Teensy serial debug output.")
    print("If values match but predictions differ, issue is in TFLite inference.")
    print("If values differ, issue is in preprocessing pipeline.")


if __name__ == '__main__':
    main()
