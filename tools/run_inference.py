#!/usr/bin/env python3
"""
Run Sleep Stage Inference on Raw EEG Data
==========================================

Flexible inference script that supports:
- Configurable input sample rates (1KHz, 4KHz, etc.)
- Configurable channel indices for bipolar EEG derivation
- Accelerometer data extraction
- Option D preprocessing (best: average downsample -> filter@100Hz)

Usage:
    # Basic usage with command line args
    python tools/run_inference.py data.bin --sample-rate 1000 --channels 12 --output predictions.csv

    # Custom bipolar derivation (CH2 - CH4)
    python tools/run_inference.py data.bin --bipolar-pos 2 --bipolar-neg 4 --output predictions.csv

    # With accelerometer extraction
    python tools/run_inference.py data.bin --accel 8 9 10 --save-accel --output predictions.csv

    # Using config file
    python tools/run_inference.py data.bin --config config.txt --output predictions.csv
"""

import numpy as np
from scipy.signal import butter, lfilter
from pathlib import Path
import argparse
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

# Default paths
DEFAULT_MODEL = Path('data/example_datasets/debug/8_tflite_quantized_model.tflite')

# Signal parameters
TARGET_SAMPLE_RATE = 100  # Hz (model input)
EPOCH_LENGTH_SEC = 30
EPOCH_SAMPLES = EPOCH_LENGTH_SEC * TARGET_SAMPLE_RATE  # 3000 samples

# Filter parameters (matching training script)
LOWCUT = 0.5   # Hz
HIGHCUT = 30   # Hz
FILTER_ORDER = 5

# Sleep stage names
STAGE_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")


def load_binary_data(filepath, num_channels, dtype=np.int32):
    """Load raw binary EEG data from file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    raw_data = np.fromfile(filepath, dtype=dtype)
    num_samples = len(raw_data) // num_channels

    if num_samples == 0:
        raise ValueError(f"No samples found in file (channels={num_channels})")

    # Reshape to (samples, channels)
    data = raw_data[:num_samples * num_channels].reshape(-1, num_channels)

    print(f"Loaded: {filepath.name}")
    print(f"  Samples: {num_samples:,}")
    print(f"  Channels: {num_channels}")
    print(f"  Shape: {data.shape}")

    return data


def extract_bipolar(data, pos_ch, neg_ch):
    """Create bipolar EEG derivation."""
    if pos_ch >= data.shape[1] or neg_ch >= data.shape[1]:
        raise ValueError(f"Channel index out of range (max: {data.shape[1]-1})")

    bipolar = data[:, pos_ch].astype(np.float64) - data[:, neg_ch].astype(np.float64)
    print(f"  Bipolar: CH{pos_ch} - CH{neg_ch}")
    print(f"  Range: [{bipolar.min():.1f}, {bipolar.max():.1f}]")

    return bipolar


def extract_accelerometer(data, x_ch, y_ch, z_ch, convert_to_g=True):
    """Extract accelerometer channels and convert to g units.

    Args:
        data: Raw data array
        x_ch, y_ch, z_ch: Channel indices for X, Y, Z axes
        convert_to_g: If True, convert raw values to g units

    Returns:
        accel_x, accel_y, accel_z in g units (if convert_to_g=True)
    """
    max_ch = data.shape[1] - 1
    if any(ch > max_ch for ch in [x_ch, y_ch, z_ch]):
        raise ValueError(f"Accelerometer channel index out of range (max: {max_ch})")

    accel_x = data[:, x_ch].astype(np.float64)
    accel_y = data[:, y_ch].astype(np.float64)
    accel_z = data[:, z_ch].astype(np.float64)

    # Convert raw ADC values to g units
    # Conversion: g = raw * 16.0 / 4095.0
    if convert_to_g:
        ACCEL_SCALE = 16.0 / 4095.0
        accel_x = accel_x * ACCEL_SCALE
        accel_y = accel_y * ACCEL_SCALE
        accel_z = accel_z * ACCEL_SCALE
        print(f"  Accelerometer: X=CH{x_ch}, Y=CH{y_ch}, Z=CH{z_ch} (converted to g)")
    else:
        print(f"  Accelerometer: X=CH{x_ch}, Y=CH{y_ch}, Z=CH{z_ch} (raw)")

    return accel_x, accel_y, accel_z


def butter_bandpass(lowcut, highcut, fs, order=5):
    """Create bandpass filter coefficients."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply bandpass filter."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def preprocess_option_d(bipolar_data, input_rate, target_rate=100):
    """
    Option D preprocessing (best results: 89.1% agreement)

    Pipeline: Input rate -> 100Hz (average downsample) -> Bandpass filter at 100Hz

    Args:
        bipolar_data: Raw bipolar EEG data at input_rate
        input_rate: Input sample rate (e.g., 1000 or 4000 Hz)
        target_rate: Target sample rate (default: 100 Hz)

    Returns:
        Preprocessed data at target_rate
    """
    print_section("Preprocessing (Option D)")

    # Calculate downsampling ratio
    ratio = input_rate // target_rate
    if input_rate % target_rate != 0:
        print(f"WARNING: Input rate {input_rate} is not evenly divisible by {target_rate}")
        ratio = int(round(input_rate / target_rate))

    print(f"  Input rate: {input_rate} Hz")
    print(f"  Target rate: {target_rate} Hz")
    print(f"  Downsample ratio: {ratio}:1")

    # Stage 1: Average downsample to target rate
    num_output_samples = len(bipolar_data) // ratio
    data_downsampled = np.zeros(num_output_samples, dtype=np.float64)

    for i in range(num_output_samples):
        start = i * ratio
        end = start + ratio
        data_downsampled[i] = np.mean(bipolar_data[start:end])

    print(f"  After downsample: {len(data_downsampled):,} samples")

    # Stage 2: Bandpass filter at target rate (0.5-30 Hz)
    data_filtered = butter_bandpass_filter(
        data_downsampled, LOWCUT, HIGHCUT, target_rate, order=FILTER_ORDER
    )

    print(f"  After bandpass filter: {len(data_filtered):,} samples")
    print(f"  Duration: {len(data_filtered) / target_rate / 3600:.2f} hours")

    return data_filtered.astype(np.float32)


def extract_and_normalize_epochs(data_100hz, num_epochs=None):
    """Extract 30-second epochs and apply Z-score normalization (per-epoch)."""
    max_epochs = len(data_100hz) // EPOCH_SAMPLES

    if num_epochs is None or num_epochs > max_epochs:
        num_epochs = max_epochs

    print(f"\nExtracting {num_epochs} epochs ({num_epochs * 30 / 60:.1f} min)...")

    epochs_normalized = np.zeros((num_epochs, EPOCH_SAMPLES), dtype=np.float32)

    for i in range(num_epochs):
        start = i * EPOCH_SAMPLES
        end = start + EPOCH_SAMPLES
        epoch = data_100hz[start:end]

        # Z-score normalization (matching training script)
        mean = np.mean(epoch)
        std = np.std(epoch)
        if std > 0:
            epochs_normalized[i] = (epoch - mean) / std
        else:
            epochs_normalized[i] = epoch - mean

    print(f"  Epochs shape: {epochs_normalized.shape}")

    return epochs_normalized


def run_tflite_inference(epochs_normalized, model_path):
    """Run inference using TFLite model."""
    if not HAS_TF:
        print("ERROR: TensorFlow not available")
        return None, None

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return None, None

    print_section("Running TFLite Inference")

    # Load model
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Model: {model_path.name}")
    print(f"  Input 0: shape={input_details[0]['shape']}, dtype={input_details[0]['dtype']}")
    print(f"  Input 1: shape={input_details[1]['shape']}, dtype={input_details[1]['dtype']}")
    print(f"  Output: shape={output_details[0]['shape']}")

    num_epochs = len(epochs_normalized)
    predictions = np.zeros(num_epochs, dtype=np.int32)
    probabilities = np.zeros((num_epochs, 5), dtype=np.float32)

    print(f"\nProcessing {num_epochs} epochs...")

    for i in range(num_epochs):
        # Prepare EEG input (shape: 1, 1, 3000, 1)
        eeg_input = epochs_normalized[i].reshape(1, 1, 3000, 1).astype(np.float32)

        # Prepare epoch index input (scaled by /1000 as in training script)
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


def save_predictions_csv(filepath, predictions, probabilities):
    """Save predictions to CSV file."""
    filepath = Path(filepath)

    with open(filepath, 'w') as f:
        f.write("epoch,timestamp_s,prob_wake,prob_n1,prob_n2,prob_n3,prob_rem,predicted_stage\n")

        for i in range(len(predictions)):
            timestamp = (i + 1) * EPOCH_LENGTH_SEC  # End time of epoch
            stage_name = STAGE_NAMES[predictions[i]]
            probs = probabilities[i]

            f.write(f"{i},{timestamp:.1f},{probs[0]:.4f},{probs[1]:.4f},"
                    f"{probs[2]:.4f},{probs[3]:.4f},{probs[4]:.4f},{stage_name}\n")

    print(f"\nSaved predictions to: {filepath}")


def save_eeg_csv(filepath, data_100hz):
    """Save preprocessed 100Hz EEG to CSV."""
    filepath = Path(filepath)

    with open(filepath, 'w') as f:
        f.write("sample_index,timestamp_ms,eeg_uv\n")
        for i, val in enumerate(data_100hz):
            timestamp_ms = i * 10  # 100Hz = 10ms per sample
            f.write(f"{i},{timestamp_ms},{val:.4f}\n")

    print(f"Saved 100Hz EEG to: {filepath}")


def save_accelerometer_csv(filepath, accel_x, accel_y, accel_z, sample_rate):
    """Save accelerometer data to CSV (in g units)."""
    filepath = Path(filepath)

    with open(filepath, 'w') as f:
        # Header indicates units are in g
        f.write("sample_index,timestamp_ms,accel_x_g,accel_y_g,accel_z_g\n")
        dt_ms = 1000.0 / sample_rate
        for i in range(len(accel_x)):
            timestamp_ms = i * dt_ms
            f.write(f"{i},{timestamp_ms:.1f},{accel_x[i]:.6f},{accel_y[i]:.6f},{accel_z[i]:.6f}\n")

    print(f"Saved accelerometer to: {filepath} (units: g)")


def print_summary(predictions):
    """Print summary of predictions."""
    print_section("Prediction Summary")

    total = len(predictions)
    print(f"Total epochs: {total}")
    print(f"Duration: {total * EPOCH_LENGTH_SEC / 3600:.2f} hours")

    print("\nStage distribution:")
    for i, name in enumerate(STAGE_NAMES):
        count = np.sum(predictions == i)
        pct = 100 * count / total if total > 0 else 0
        time_min = count * EPOCH_LENGTH_SEC / 60
        print(f"  {name:5s}: {count:4d} epochs ({pct:5.1f}%) = {time_min:.1f} min")


def load_config(config_path):
    """Load configuration from file."""
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Run sleep stage inference on raw EEG data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required
    parser.add_argument('data_file', help='Input binary EEG file')

    # Configuration options
    parser.add_argument('--config', '-c', help='Config file (alternative to CLI args)')
    parser.add_argument('--output', '-o', help='Output predictions CSV')

    # Data format
    parser.add_argument('--sample-rate', '-r', type=int, default=1000,
                        help='Input sample rate in Hz (default: 1000)')
    parser.add_argument('--channels', '-n', type=int, default=12,
                        help='Number of channels in file (default: 12)')
    parser.add_argument('--dtype', choices=['int32', 'int16', 'float32'], default='int32',
                        help='Data type (default: int32)')

    # EEG channels
    parser.add_argument('--bipolar-pos', type=int, default=0,
                        help='Positive electrode channel index (default: 0)')
    parser.add_argument('--bipolar-neg', type=int, default=6,
                        help='Negative electrode channel index (default: 6)')

    # Accelerometer
    parser.add_argument('--accel', nargs=3, type=int, metavar=('X', 'Y', 'Z'),
                        help='Accelerometer channel indices (e.g., --accel 8 9 10)')

    # Model
    parser.add_argument('--model', '-m', type=str,
                        help=f'TFLite model path (default: {DEFAULT_MODEL})')

    # Output options
    parser.add_argument('--save-eeg', action='store_true',
                        help='Save preprocessed 100Hz EEG to CSV')
    parser.add_argument('--save-accel', action='store_true',
                        help='Save accelerometer data to CSV')

    # Processing options
    parser.add_argument('--max-epochs', type=int,
                        help='Maximum number of epochs to process')

    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        # Override args with config values
        if 'sample_rate' in config:
            args.sample_rate = int(config['sample_rate'])
        if 'channels' in config:
            args.channels = int(config['channels'])
        if 'bipolar_channel_positive' in config:
            args.bipolar_pos = int(config['bipolar_channel_positive'])
        if 'bipolar_channel_negative' in config:
            args.bipolar_neg = int(config['bipolar_channel_negative'])
        if 'accel_channel_x' in config and args.accel is None:
            args.accel = [
                int(config['accel_channel_x']),
                int(config['accel_channel_y']),
                int(config['accel_channel_z'])
            ]

    # Set defaults
    if args.model is None:
        args.model = str(DEFAULT_MODEL)

    if args.output is None:
        data_path = Path(args.data_file)
        args.output = data_path.stem + '_predictions.csv'

    # Print configuration
    print_section("Configuration")
    print(f"  Data file: {args.data_file}")
    print(f"  Sample rate: {args.sample_rate} Hz")
    print(f"  Channels: {args.channels}")
    print(f"  Bipolar: CH{args.bipolar_pos} - CH{args.bipolar_neg}")
    if args.accel:
        print(f"  Accelerometer: X=CH{args.accel[0]}, Y=CH{args.accel[1]}, Z=CH{args.accel[2]}")
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output}")

    # Map dtype string to numpy dtype
    dtype_map = {'int32': np.int32, 'int16': np.int16, 'float32': np.float32}
    dtype = dtype_map[args.dtype]

    # Load data
    print_section("Loading Data")
    data = load_binary_data(args.data_file, args.channels, dtype=dtype)

    # Extract bipolar EEG
    bipolar = extract_bipolar(data, args.bipolar_pos, args.bipolar_neg)

    # Extract accelerometer if requested
    accel_data = None
    if args.accel:
        accel_data = extract_accelerometer(data, args.accel[0], args.accel[1], args.accel[2])

    # Preprocess EEG
    data_100hz = preprocess_option_d(bipolar, args.sample_rate, TARGET_SAMPLE_RATE)

    # Extract and normalize epochs
    epochs = extract_and_normalize_epochs(data_100hz, args.max_epochs)

    # Run inference
    predictions, probabilities = run_tflite_inference(epochs, args.model)

    if predictions is not None:
        # Print summary
        print_summary(predictions)

        # Save predictions
        save_predictions_csv(args.output, predictions, probabilities)

        # Save preprocessed EEG if requested
        if args.save_eeg:
            eeg_output = Path(args.output).stem.replace('_predictions', '') + '_eeg_100hz.csv'
            save_eeg_csv(eeg_output, data_100hz)

        # Save accelerometer if requested
        if args.save_accel and accel_data is not None:
            accel_output = Path(args.output).stem.replace('_predictions', '') + '_accel.csv'
            save_accelerometer_csv(accel_output, accel_data[0], accel_data[1], accel_data[2],
                                   args.sample_rate)

    print("\nDone!")


if __name__ == '__main__':
    main()
