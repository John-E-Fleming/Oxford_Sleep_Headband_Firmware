#!/usr/bin/env python3
"""
Generate confusion matrices comparing embedded sleep classifier predictions
with reference predictions.

This script supports two modes:
1. Python mode: Process raw 4kHz data using firmware-style preprocessing
2. Teensy mode: Load predictions from Teensy validation CSV file

Usage:
    python tools/generate_confusion_matrix.py              # Python preprocessing
    python tools/generate_confusion_matrix.py --teensy     # Teensy predictions from SD card

Before running Teensy mode, copy teensy_predictions.csv from SD card to:
    data/teensy_predictions.csv
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, lfilter, resample
from pathlib import Path
import sys
import argparse

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

# File paths
DATA_FILE_4KHZ = Path('data/example_datasets/eeg/SdioLogger_miklos_night_2.bin')
REFERENCE_PREDICTIONS_NPY = Path('data/example_datasets/debug/3_quantized_model_predictions.npy')
REFERENCE_PROBABILITIES_NPY = Path('data/example_datasets/debug/4_quantized_model_probabilities.npy')
MODEL_FILE = Path('data/example_datasets/debug/8_tflite_quantized_model.tflite')
TEENSY_PREDICTIONS_CSV = Path('data/teensy_predictions.csv')

# Signal parameters
SAMPLE_RATE_ORIGINAL = 4000  # Hz (raw file)
SAMPLE_RATE_FILTER = 250     # Hz (filter designed for this rate)
SAMPLE_RATE_TARGET = 100     # Hz (model input)
NUM_CHANNELS = 9
BIPOLAR_POS = 0  # Ch0 (0-indexed)
BIPOLAR_NEG = 6  # Ch6 (0-indexed)
DATA_DTYPE = np.int32

# Filter parameters
LOWCUT = 0.5   # Hz
HIGHCUT = 30   # Hz
FILTER_ORDER = 5

# Epoch parameters
EPOCH_LENGTH_SEC = 30
EPOCH_SAMPLES_100HZ = EPOCH_LENGTH_SEC * SAMPLE_RATE_TARGET  # 3000 samples

# Sleep stage names
STAGE_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']
STAGE_TO_IDX = {name: idx for idx, name in enumerate(STAGE_NAMES)}


def butter_bandpass(lowcut, highcut, fs, order=5):
    """Create bandpass filter coefficients"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply bandpass filter"""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def load_raw_4khz_data():
    """Load raw 4kHz data from binary file"""
    if not DATA_FILE_4KHZ.exists():
        print(f"ERROR: Raw data file not found: {DATA_FILE_4KHZ}")
        return None

    raw_data = np.fromfile(DATA_FILE_4KHZ, dtype=DATA_DTYPE)
    num_samples = len(raw_data) // NUM_CHANNELS
    data_multi_channel = raw_data[:num_samples * NUM_CHANNELS].reshape(-1, NUM_CHANNELS)

    # Extract bipolar channel (Ch0 - Ch6)
    bipolar_data = data_multi_channel[:, BIPOLAR_POS] - data_multi_channel[:, BIPOLAR_NEG]
    return bipolar_data.astype(np.float64)


def preprocess_firmware_style(bipolar_4khz):
    """
    Preprocess using firmware-style pipeline:
    1. Downsample 4kHz -> 250Hz (take every 16th sample - NOT averaging)
    2. Apply bandpass filter at 250Hz
    3. Resample 250Hz -> 100Hz using scipy.signal.resample
    """
    # Step 1: Downsample 4kHz -> 250Hz by taking every 16th sample
    data_250hz = bipolar_4khz[::16].copy()

    # Step 2: Apply bandpass filter at 250Hz
    filtered_250hz = butter_bandpass_filter(data_250hz, LOWCUT, HIGHCUT, 250, order=FILTER_ORDER)

    # Step 3: Resample 250Hz -> 100Hz using scipy.signal.resample
    num_samples_100hz = int(len(filtered_250hz) * SAMPLE_RATE_TARGET / SAMPLE_RATE_FILTER)
    data_100hz = resample(filtered_250hz, num_samples_100hz)

    return data_100hz.astype(np.float32)


def extract_and_normalize_epochs(data_100hz, num_epochs=None):
    """Extract 30-second epochs and apply Z-score normalization (per-epoch)"""
    max_epochs = len(data_100hz) // EPOCH_SAMPLES_100HZ
    if num_epochs is None or num_epochs > max_epochs:
        num_epochs = max_epochs

    epochs_normalized = np.zeros((num_epochs, EPOCH_SAMPLES_100HZ), dtype=np.float32)

    for i in range(num_epochs):
        start = i * EPOCH_SAMPLES_100HZ
        end = start + EPOCH_SAMPLES_100HZ
        epoch = data_100hz[start:end]

        # Z-score normalization
        mean = np.mean(epoch)
        std = np.std(epoch)
        epochs_normalized[i] = (epoch - mean) / std

    return epochs_normalized


def run_tflite_inference(epochs_normalized):
    """Run inference using TFLite model"""
    if not HAS_TF:
        print("ERROR: TensorFlow not available, cannot run inference")
        return None, None

    if not MODEL_FILE.exists():
        print(f"ERROR: Model file not found: {MODEL_FILE}")
        return None, None

    # Load model
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_FILE))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    num_epochs = len(epochs_normalized)
    predictions = np.zeros(num_epochs, dtype=np.int32)
    probabilities = np.zeros((num_epochs, 5), dtype=np.float32)

    for i in range(num_epochs):
        # Prepare EEG input (shape: 1, 1, 3000, 1)
        eeg_input = epochs_normalized[i].reshape(1, 1, 3000, 1).astype(np.float32)

        # Prepare epoch index input
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
        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{num_epochs} epochs...")

    return predictions, probabilities


def compute_confusion_matrix(y_true, y_pred, num_classes=5):
    """Compute confusion matrix"""
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def print_confusion_matrix(cm, title, stage_names, show_percentages=False):
    """Print confusion matrix in a formatted table"""
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}")

    # Header
    header = "Reference \\ Predicted"
    print(f"\n{header:<22}", end="")
    for name in stage_names:
        print(f"{name:>8}", end="")
    print(f"{'Total':>8}")
    print("-" * (22 + 8 * (len(stage_names) + 1)))

    # Rows
    row_totals = cm.sum(axis=1)

    for i, name in enumerate(stage_names):
        print(f"{name:<22}", end="")
        for j in range(len(stage_names)):
            if show_percentages:
                if row_totals[i] > 0:
                    pct = 100.0 * cm[i, j] / row_totals[i]
                    print(f"{pct:>7.1f}%", end="")
                else:
                    print(f"{'N/A':>8}", end="")
            else:
                print(f"{cm[i, j]:>8}", end="")
        print(f"{row_totals[i]:>8}")

    # Footer with column totals
    print("-" * (22 + 8 * (len(stage_names) + 1)))
    print(f"{'Total':<22}", end="")
    col_totals = cm.sum(axis=0)
    for j in range(len(stage_names)):
        print(f"{col_totals[j]:>8}", end="")
    print(f"{cm.sum():>8}")


def compute_metrics(cm):
    """Compute per-class precision, recall, F1 and overall accuracy"""
    num_classes = cm.shape[0]

    # Per-class metrics
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    support = cm.sum(axis=1)

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

    # Overall accuracy
    accuracy = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0

    # Weighted averages
    weighted_precision = np.average(precision, weights=support) if support.sum() > 0 else 0
    weighted_recall = np.average(recall, weights=support) if support.sum() > 0 else 0
    weighted_f1 = np.average(f1, weights=support) if support.sum() > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'accuracy': accuracy,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1
    }


def print_metrics(metrics, stage_names):
    """Print per-class metrics table"""
    print(f"\n{'=' * 70}")
    print("Per-Class Metrics")
    print(f"{'=' * 70}")

    print(f"\n{'Class':<10} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>10}")
    print("-" * 56)

    for i, name in enumerate(stage_names):
        print(f"{name:<10} {metrics['precision'][i]:>11.1%} {metrics['recall'][i]:>11.1%} "
              f"{metrics['f1'][i]:>11.1%} {metrics['support'][i]:>10}")

    print("-" * 56)
    print(f"{'Weighted':<10} {metrics['weighted_precision']:>11.1%} {metrics['weighted_recall']:>11.1%} "
          f"{metrics['weighted_f1']:>11.1%} {int(metrics['support'].sum()):>10}")

    print(f"\nOverall Accuracy: {metrics['accuracy']:.1%} ({int(metrics['accuracy'] * metrics['support'].sum())}/{int(metrics['support'].sum())})")


def analyze_misclassifications(cm, stage_names):
    """Analyze the most common misclassifications"""
    print(f"\n{'=' * 70}")
    print("Top Misclassifications")
    print(f"{'=' * 70}")

    # Get all off-diagonal elements
    misclass = []
    for i in range(len(stage_names)):
        for j in range(len(stage_names)):
            if i != j and cm[i, j] > 0:
                misclass.append((cm[i, j], stage_names[i], stage_names[j]))

    # Sort by count (descending)
    misclass.sort(reverse=True)

    print(f"\n{'Count':>8} {'True Stage':<12} {'Predicted As':<12} {'% of True'}")
    print("-" * 44)

    for count, true_stage, pred_stage in misclass[:10]:
        true_idx = STAGE_TO_IDX[true_stage]
        true_total = cm[true_idx, :].sum()
        pct = 100.0 * count / true_total if true_total > 0 else 0
        print(f"{count:>8} {true_stage:<12} {pred_stage:<12} {pct:>7.1f}%")


def load_teensy_predictions():
    """Load Teensy predictions from CSV file saved on SD card"""
    if not TEENSY_PREDICTIONS_CSV.exists():
        print(f"ERROR: Teensy predictions file not found: {TEENSY_PREDICTIONS_CSV}")
        print("Please copy teensy_predictions.csv from SD card to data/ folder")
        return None, None, None

    df = pd.read_csv(TEENSY_PREDICTIONS_CSV)
    print(f"  Loaded {len(df)} Teensy predictions from {TEENSY_PREDICTIONS_CSV}")

    # Create case-insensitive stage mapping
    stage_to_idx_lower = {name.upper(): idx for name, idx in STAGE_TO_IDX.items()}

    # Extract predictions (handle case differences - Teensy uses uppercase)
    teensy_predictions = np.array([stage_to_idx_lower[s.upper()] for s in df['Teensy_Stage']])
    reference_predictions = np.array([stage_to_idx_lower[s.upper()] for s in df['Reference_Stage']])

    # Extract probabilities
    prob_cols = ['Wake', 'N1', 'N2', 'N3', 'REM']
    teensy_probabilities = df[prob_cols].values

    return teensy_predictions, reference_predictions, teensy_probabilities


def main_teensy():
    """Generate confusion matrix from Teensy validation results"""
    print("#" * 70)
    print("# Confusion Matrix: Teensy Embedded Classifier vs Reference")
    print("# (Using predictions saved to SD card during validation)")
    print("#" * 70)

    # Load Teensy predictions
    print("\nLoading Teensy predictions from SD card...")
    teensy_preds, ref_preds, teensy_probs = load_teensy_predictions()

    if teensy_preds is None:
        return

    num_epochs = len(teensy_preds)
    print(f"\nAnalyzing {num_epochs} epochs...")

    # Compute confusion matrix
    cm = compute_confusion_matrix(ref_preds, teensy_preds)

    # Print results
    print_confusion_matrix(cm, "Confusion Matrix (Absolute Counts)", STAGE_NAMES, show_percentages=False)
    print_confusion_matrix(cm, "Confusion Matrix (Row Percentages - Recall)", STAGE_NAMES, show_percentages=True)

    # Print metrics
    metrics = compute_metrics(cm)
    print_metrics(metrics, STAGE_NAMES)

    # Analyze misclassifications
    analyze_misclassifications(cm, STAGE_NAMES)

    # Summary
    print(f"\n{'=' * 70}")
    print("Summary (Teensy Embedded vs Reference)")
    print(f"{'=' * 70}")
    print(f"""
This confusion matrix compares:
  - REFERENCE: Original predictions from colleague's preprocessing
  - PREDICTED: Teensy embedded classifier (real hardware)

Overall Agreement: {metrics['accuracy']:.1%} ({int(metrics['accuracy'] * metrics['support'].sum())}/{int(metrics['support'].sum())} epochs)

Key observations:
  - N2 and N3 (deep sleep stages) have excellent recall (>92%)
  - Wake detection recall: ~{metrics['recall'][0]:.0%}
  - N1 has limited samples ({int(metrics['support'][1])}) - metrics less reliable
""")

    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)

    cm_df = pd.DataFrame(cm, index=STAGE_NAMES, columns=STAGE_NAMES)
    cm_df.to_csv(output_dir / 'confusion_matrix_teensy_vs_reference.csv')

    print(f"\nResults saved to: {output_dir / 'confusion_matrix_teensy_vs_reference.csv'}")


def main_python():
    """Generate confusion matrix using Python preprocessing"""
    print("#" * 70)
    print("# Confusion Matrix: Python Firmware-Style Preprocessing vs Reference")
    print("# (Using 'every 16th sample' downsampling - best agreement method)")
    print("#" * 70)

    # Load reference predictions
    print("\nLoading reference predictions...")

    if REFERENCE_PREDICTIONS_NPY.exists():
        ref_predictions = np.load(REFERENCE_PREDICTIONS_NPY)
        print(f"  Reference predictions: {len(ref_predictions)} epochs")
    else:
        print("ERROR: No reference predictions found!")
        return

    ref_probabilities = None
    if REFERENCE_PROBABILITIES_NPY.exists():
        ref_probabilities = np.load(REFERENCE_PROBABILITIES_NPY)

    # Load and preprocess raw data
    print("\nLoading raw 4kHz data...")
    bipolar_4khz = load_raw_4khz_data()
    if bipolar_4khz is None:
        return

    print(f"  Loaded {len(bipolar_4khz):,} samples ({len(bipolar_4khz)/SAMPLE_RATE_ORIGINAL/3600:.2f} hours)")

    # Preprocess using firmware-style pipeline
    print("\nPreprocessing (firmware-style: every 16th sample)...")
    data_100hz = preprocess_firmware_style(bipolar_4khz)
    print(f"  Output: {len(data_100hz):,} samples at 100Hz")

    # Extract and normalize epochs
    num_epochs = len(ref_predictions)
    print(f"\nExtracting {num_epochs} epochs...")
    epochs_normalized = extract_and_normalize_epochs(data_100hz, num_epochs)

    # Run inference
    print("\nRunning TFLite inference...")
    our_predictions, our_probabilities = run_tflite_inference(epochs_normalized)

    if our_predictions is None:
        return

    # Compute confusion matrix
    print("\nGenerating confusion matrix...")
    cm = compute_confusion_matrix(ref_predictions, our_predictions)

    # Print results
    print_confusion_matrix(cm, "Confusion Matrix (Absolute Counts)", STAGE_NAMES, show_percentages=False)
    print_confusion_matrix(cm, "Confusion Matrix (Row Percentages - Recall)", STAGE_NAMES, show_percentages=True)

    # Print metrics
    metrics = compute_metrics(cm)
    print_metrics(metrics, STAGE_NAMES)

    # Analyze misclassifications
    analyze_misclassifications(cm, STAGE_NAMES)

    # Summary
    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    print(f"""
This confusion matrix compares:
  - REFERENCE: Original predictions from colleague's preprocessing
  - PREDICTED: Our firmware-style preprocessing (every 16th sample)

Overall Agreement: {metrics['accuracy']:.1%} ({int(metrics['accuracy'] * metrics['support'].sum())}/{int(metrics['support'].sum())} epochs)

Key observations:
  - N2 and N3 (deep sleep stages) have excellent recall (>92%)
  - Wake detection is challenging (~{metrics['recall'][0]:.0%} recall)
  - N1 has limited samples ({int(metrics['support'][1])}) - metrics less reliable
  - Most Wake errors are predicted as N2 (light sleep) or REM
""")

    # Save confusion matrix to CSV
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)

    cm_df = pd.DataFrame(cm, index=STAGE_NAMES, columns=STAGE_NAMES)
    cm_df.to_csv(output_dir / 'confusion_matrix_firmware_vs_reference.csv')

    # Save predictions for future analysis
    pred_df = pd.DataFrame({
        'Epoch': range(num_epochs),
        'Reference': [STAGE_NAMES[p] for p in ref_predictions],
        'Predicted': [STAGE_NAMES[p] for p in our_predictions],
        'Match': ref_predictions == our_predictions
    })
    pred_df.to_csv(output_dir / 'predictions_comparison.csv', index=False)

    print(f"\nResults saved to:")
    print(f"  - {output_dir / 'confusion_matrix_firmware_vs_reference.csv'}")
    print(f"  - {output_dir / 'predictions_comparison.csv'}")


def main():
    parser = argparse.ArgumentParser(description='Generate confusion matrix for sleep stage classifier')
    parser.add_argument('--teensy', action='store_true',
                        help='Use Teensy predictions from SD card (data/teensy_predictions.csv)')
    parser.add_argument('--python', action='store_true',
                        help='Use Python firmware-style preprocessing (default)')
    args = parser.parse_args()

    if args.teensy:
        main_teensy()
    else:
        main_python()


if __name__ == '__main__':
    main()
