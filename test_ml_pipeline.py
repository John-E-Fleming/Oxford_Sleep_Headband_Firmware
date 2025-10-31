#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML Pipeline Testing Script for PC-based Debugging
==================================================

This script tests the sleep headband ML preprocessing and inference pipeline
by processing reference data from a working implementation and comparing
step-by-step outputs.

Usage:
    python test_ml_pipeline.py

The script will:
1. Load the bandpassed EEG data (100Hz, 0.5-30Hz filtered)
2. Test per-epoch z-score normalization (as done in reference)
3. Quantize to INT8 for model input
4. Run inference with TFLite model
5. Compare all outputs with reference data
6. Generate detailed reports showing where differences occur

Reference files (in data/example_datasets/debug/):
- 1_bandpassed_eeg_single_channel.npy: Raw bandpassed EEG @ 100Hz
- 2_standardized_epochs.npy: Z-score normalized 30s epochs
- 3_quantized_model_predictions.npy: Model predictions (class indices)
- 4_quantized_model_probabilities.npy: Model output probabilities
- 8_tflite_quantized_model.tflite: The TFLite model
"""

import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Model constants (matching model.h in firmware)
MODEL_INPUT_SIZE = 3001  # 3000 EEG samples + 1 epoch index
MODEL_EEG_SAMPLES = 3000  # 30 seconds at 100Hz
MODEL_OUTPUT_SIZE = 5     # 5 classes: N3, N2, N1, REM, Wake
ML_SAMPLE_RATE = 100      # 100Hz

# Sleep stage names (matching model.h enum)
SLEEP_STAGE_NAMES = ['N3_Deep', 'N2_Light', 'N1_VeryLight', 'REM', 'Wake']

# Reference data directory
DEBUG_DIR = Path('data/example_datasets/debug')


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}")


def print_subsection(title):
    """Print a formatted subsection header"""
    print(f"\n{'-' * 70}")
    print(f"{title}")
    print(f"{'-' * 70}")


def load_reference_data():
    """Load all reference files"""
    print_section("Loading Reference Data")

    files = {
        'eeg': DEBUG_DIR / '1_bandpassed_eeg_single_channel.npy',
        'normalized': DEBUG_DIR / '2_standardized_epochs.npy',
        'predictions': DEBUG_DIR / '3_quantized_model_predictions.npy',
        'probabilities': DEBUG_DIR / '4_quantized_model_probabilities.npy',
        'model': DEBUG_DIR / '8_tflite_quantized_model.tflite'
    }

    data = {}

    for name, filepath in files.items():
        if name == 'model':
            if filepath.exists():
                print(f"✓ Found model: {filepath}")
                data[name] = str(filepath)
            else:
                print(f"⚠ Model not found: {filepath}")
                data[name] = None
        else:
            if filepath.exists():
                data[name] = np.load(filepath)
                print(f"✓ Loaded {name}: {filepath}")
                print(f"  Shape: {data[name].shape}")
                if name == 'eeg':
                    print(f"  Duration: {len(data[name]) / ML_SAMPLE_RATE / 3600:.2f} hours")
            else:
                raise FileNotFoundError(f"Required file not found: {filepath}")

    return data


def test_normalization(ref_data, num_epochs=10):
    """
    Test per-epoch z-score normalization

    The reference implementation normalizes each 30-second epoch independently:
    normalized = (epoch - epoch_mean) / epoch_std

    This matches what should be done in firmware for proper ML inference.
    """
    print_section("Testing Normalization")

    eeg_data = ref_data['eeg']
    ref_normalized = ref_data['normalized']

    num_epochs = min(num_epochs, ref_normalized.shape[0])

    max_differences = []
    mean_differences = []

    for epoch_idx in range(num_epochs):
        print_subsection(f"Epoch {epoch_idx} (Time: {epoch_idx * 30}s - {(epoch_idx + 1) * 30}s)")

        # Extract raw epoch
        start_idx = epoch_idx * MODEL_EEG_SAMPLES
        end_idx = start_idx + MODEL_EEG_SAMPLES
        raw_epoch = eeg_data[start_idx:end_idx]

        # Compute per-epoch statistics
        epoch_mean = raw_epoch.mean()
        epoch_std = raw_epoch.std()

        print(f"Raw epoch statistics:")
        print(f"  Mean: {epoch_mean:.6f}")
        print(f"  Std:  {epoch_std:.6f}")
        print(f"  Range: [{raw_epoch.min():.2f}, {raw_epoch.max():.2f}]")

        # Normalize using per-epoch z-score
        our_normalized = (raw_epoch - epoch_mean) / epoch_std

        print(f"Our normalized epoch:")
        print(f"  Mean: {our_normalized.mean():.6f}")
        print(f"  Std:  {our_normalized.std():.6f}")
        print(f"  Range: [{our_normalized.min():.2f}, {our_normalized.max():.2f}]")

        # Get reference normalized epoch
        ref_epoch = ref_normalized[epoch_idx]

        print(f"Reference normalized epoch:")
        print(f"  Mean: {ref_epoch.mean():.6f}")
        print(f"  Std:  {ref_epoch.std():.6f}")
        print(f"  Range: [{ref_epoch.min():.2f}, {ref_epoch.max():.2f}]")

        # Compare
        differences = np.abs(our_normalized - ref_epoch)
        max_diff = differences.max()
        mean_diff = differences.mean()
        max_diff_idx = differences.argmax()

        max_differences.append(max_diff)
        mean_differences.append(mean_diff)

        print(f"\nComparison:")
        print(f"  Max difference:  {max_diff:.8f} at index {max_diff_idx}")
        print(f"  Mean difference: {mean_diff:.8f}")

        if mean_diff < 1e-4:
            print(f"  ✓ PASS: Normalization matches reference")
        else:
            print(f"  ✗ FAIL: Normalization differs from reference")
            print(f"\n  First 10 sample comparison:")
            for i in range(min(10, len(our_normalized))):
                print(f"    [{i:4d}] Ours: {our_normalized[i]:10.6f} | Ref: {ref_epoch[i]:10.6f} | Diff: {differences[i]:.8f}")

    # Summary
    print_subsection("Normalization Test Summary")
    print(f"Epochs tested: {num_epochs}")
    print(f"Max difference (overall):  {max(max_differences):.8f}")
    print(f"Mean difference (overall): {np.mean(mean_differences):.8f}")

    if all(d < 1e-4 for d in mean_differences):
        print("✓ ALL TESTS PASSED: Normalization implementation is correct!")
    else:
        print("✗ SOME TESTS FAILED: Normalization implementation differs from reference")

    return mean_differences


def test_inference(ref_data, num_epochs=10):
    """
    Test ML inference with TFLite model
    """
    print_section("Testing ML Inference")

    model_path = ref_data['model']
    if model_path is None:
        print("⚠ Model file not found, skipping inference test")
        return None

    # Try to load TFLite
    try:
        import tensorflow as tf
    except ImportError:
        print("⚠ TensorFlow not installed, skipping inference test")
        print("  Install with: pip install tensorflow")
        return None

    # Load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"✓ Model loaded: {model_path}")
    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Input dtype: {input_details[0]['dtype']}")
    print(f"  Input scale: {input_details[0]['quantization'][0]}")
    print(f"  Input zero_point: {input_details[0]['quantization'][1]}")
    print(f"  Output shape: {output_details[0]['shape']}")
    print(f"  Output dtype: {output_details[0]['dtype']}")
    print(f"  Output scale: {output_details[0]['quantization'][0]}")
    print(f"  Output zero_point: {output_details[0]['quantization'][1]}")

    ref_normalized = ref_data['normalized']
    ref_predictions = ref_data['predictions']
    ref_probabilities = ref_data['probabilities']

    num_epochs = min(num_epochs, ref_normalized.shape[0])

    predictions_matched = 0
    max_prob_diffs = []
    mean_prob_diffs = []

    # Store all predictions and probabilities
    all_our_predictions = []
    all_our_probabilities = []

    # Check if model uses quantization
    input_scale = input_details[0]['quantization'][0]
    input_zero_point = input_details[0]['quantization'][1]
    is_quantized = (input_scale != 0.0)

    output_scale = output_details[0]['quantization'][0]
    output_zero_point = output_details[0]['quantization'][1]
    output_is_quantized = (output_scale != 0.0)

    print(f"\nProcessing {num_epochs} epochs...")
    if not is_quantized:
        print("NOTE: Model uses FLOAT32 inputs (not quantized INT8)")

    for epoch_idx in range(num_epochs):
        # Print progress every 50 epochs or for first 3
        verbose = (epoch_idx < 3) or (epoch_idx % 50 == 0) or (epoch_idx == num_epochs - 1)

        if verbose or (epoch_idx % 50 == 0):
            print(f"\nProcessing epoch {epoch_idx}/{num_epochs} ({100*epoch_idx/num_epochs:.1f}%)")

        # Get reference normalized epoch
        ref_epoch = ref_normalized[epoch_idx]

        # Prepare input (3000 samples + epoch index)
        input_data = np.zeros(MODEL_INPUT_SIZE, dtype=np.float32)
        input_data[:MODEL_EEG_SAMPLES] = ref_epoch
        input_data[MODEL_EEG_SAMPLES] = float(epoch_idx)

        if is_quantized:
            # Quantize input
            input_quantized = np.clip(
                np.round(input_data / input_scale + input_zero_point),
                -128, 127
            ).astype(np.int8)
            interpreter.set_tensor(input_details[0]['index'], input_quantized.reshape(1, -1))
        else:
            # Model expects float32 input directly
            if input_details[0]['shape'].tolist() == [1, 1, 3000, 1]:
                eeg_input = ref_epoch.reshape(1, 1, 3000, 1).astype(np.float32)
                interpreter.set_tensor(input_details[0]['index'], eeg_input)
            else:
                interpreter.set_tensor(input_details[0]['index'], input_data.reshape(1, -1))

        # Run inference
        interpreter.invoke()

        # Get output
        if output_is_quantized:
            output_quantized = interpreter.get_tensor(output_details[0]['index'])[0]
            our_probs = (output_quantized.astype(np.float32) - output_zero_point) * output_scale
        else:
            our_probs = interpreter.get_tensor(output_details[0]['index'])[0]

        # Get predictions
        our_pred = np.argmax(our_probs)
        ref_pred = ref_predictions[epoch_idx]
        ref_probs = ref_probabilities[epoch_idx]

        # Store predictions and probabilities
        all_our_predictions.append(our_pred)
        all_our_probabilities.append(our_probs)

        # Compare
        pred_match = (our_pred == ref_pred)
        prob_diffs = np.abs(our_probs - ref_probs)
        max_prob_diff = prob_diffs.max()
        mean_prob_diff = prob_diffs.mean()

        if pred_match:
            predictions_matched += 1

        max_prob_diffs.append(max_prob_diff)
        mean_prob_diffs.append(mean_prob_diff)

        # Show details for verbose epochs or mismatches
        if verbose or not pred_match:
            print(f"  Epoch {epoch_idx} - Ref: {SLEEP_STAGE_NAMES[ref_pred]:15s}, Ours: {SLEEP_STAGE_NAMES[our_pred]:15s}, Match: {'✓' if pred_match else '✗'}, Prob diff: {mean_prob_diff:.6f}")

    # Convert to numpy arrays
    all_our_predictions = np.array(all_our_predictions)
    all_our_probabilities = np.array(all_our_probabilities)

    # Summary
    print_subsection("Inference Test Summary")
    print(f"Epochs tested: {num_epochs}")
    print(f"Predictions matched: {predictions_matched} / {num_epochs} ({100 * predictions_matched / num_epochs:.1f}%)")
    print(f"Max probability difference (overall):  {max(max_prob_diffs):.6f}")
    print(f"Mean probability difference (overall): {np.mean(mean_prob_diffs):.6f}")

    if predictions_matched == num_epochs:
        print("✓ ALL PREDICTIONS MATCH!")
    else:
        print(f"✗ {num_epochs - predictions_matched} PREDICTIONS DIFFER")

    return predictions_matched, mean_prob_diffs, all_our_predictions, all_our_probabilities


def analyze_mismatches(ref_predictions, our_predictions, ref_probabilities, our_probabilities, output_dir='debug_outputs'):
    """
    Analyze prediction mismatches and generate confusion matrix
    """
    print_section("Mismatch Analysis")

    # Create confusion matrix
    confusion_matrix = np.zeros((MODEL_OUTPUT_SIZE, MODEL_OUTPUT_SIZE), dtype=int)
    for ref, ours in zip(ref_predictions, our_predictions):
        confusion_matrix[ref, ours] += 1

    print("\nConfusion Matrix:")
    print("Rows = Reference (True), Columns = Our Predictions")
    print("\n" + " " * 18 + "  ".join([f"{name[:4]:>6s}" for name in SLEEP_STAGE_NAMES]))
    for i, row_name in enumerate(SLEEP_STAGE_NAMES):
        print(f"{row_name:15s} | " + "  ".join([f"{confusion_matrix[i, j]:6d}" for j in range(MODEL_OUTPUT_SIZE)]))

    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for i, name in enumerate(SLEEP_STAGE_NAMES):
        total = confusion_matrix[i, :].sum()
        correct = confusion_matrix[i, i]
        if total > 0:
            accuracy = 100.0 * correct / total
            print(f"  {name:15s}: {correct:3d}/{total:3d} ({accuracy:.1f}%)")
        else:
            print(f"  {name:15s}: No samples")

    # Find most common misclassifications
    print("\nMost Common Misclassifications:")
    misclass_counts = []
    for i in range(MODEL_OUTPUT_SIZE):
        for j in range(MODEL_OUTPUT_SIZE):
            if i != j and confusion_matrix[i, j] > 0:
                misclass_counts.append((confusion_matrix[i, j], i, j))

    misclass_counts.sort(reverse=True)
    for count, true_class, pred_class in misclass_counts[:10]:
        print(f"  {SLEEP_STAGE_NAMES[true_class]:15s} → {SLEEP_STAGE_NAMES[pred_class]:15s}: {count:3d} times")

    # Analyze probability differences
    prob_diffs = np.abs(our_probabilities - ref_probabilities)

    print("\nProbability Difference Statistics:")
    print(f"  Mean difference:    {prob_diffs.mean():.6f}")
    print(f"  Median difference:  {np.median(prob_diffs):.6f}")
    print(f"  Max difference:     {prob_diffs.max():.6f}")
    print(f"  Std deviation:      {prob_diffs.std():.6f}")

    # Save confusion matrix
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'confusion_matrix.npy'), confusion_matrix)

    # Save mismatch indices
    mismatch_indices = np.where(ref_predictions != our_predictions)[0]
    np.save(os.path.join(output_dir, 'mismatch_indices.npy'), mismatch_indices)

    print(f"\n✓ Saved confusion matrix and mismatch indices to {output_dir}/")

    return confusion_matrix, mismatch_indices


def visualize_results(ref_predictions, our_predictions, ref_probabilities, our_probabilities,
                      eeg_data=None, output_dir='debug_outputs'):
    """
    Create hypnogram and probability heatmap visualization
    """
    print_section("Creating Visualizations")

    num_epochs = len(ref_predictions)
    time_hours = np.arange(num_epochs) * 30 / 3600  # Convert epochs to hours

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 2, 2], hspace=0.3)

    # Sleep stage colors (matching typical sleep visualization)
    stage_colors = {
        0: '#2E4057',  # N3 - Dark blue (deep sleep)
        1: '#048A81',  # N2 - Teal (light sleep)
        2: '#54C6EB',  # N1 - Light blue (very light sleep)
        3: '#8E7DBE',  # REM - Purple
        4: '#F18F01'   # Wake - Orange
    }

    # 1. Reference Hypnogram
    ax1 = fig.add_subplot(gs[0])
    for i in range(num_epochs):
        color = stage_colors[ref_predictions[i]]
        ax1.barh(0, 30/3600, left=time_hours[i], height=0.8, color=color, edgecolor='none')
    ax1.set_xlim(0, time_hours[-1])
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_yticks([0])
    ax1.set_yticklabels(['Reference'])
    ax1.set_xlabel('')
    ax1.set_title('Reference Sleep Stages (from Colleague\'s Working Implementation)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # 2. Our Hypnogram
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    for i in range(num_epochs):
        color = stage_colors[our_predictions[i]]
        # Highlight mismatches with red border
        edgecolor = 'red' if our_predictions[i] != ref_predictions[i] else 'none'
        linewidth = 1.5 if our_predictions[i] != ref_predictions[i] else 0
        ax2.barh(0, 30/3600, left=time_hours[i], height=0.8, color=color,
                edgecolor=edgecolor, linewidth=linewidth)
    ax2.set_xlim(0, time_hours[-1])
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_yticks([0])
    ax2.set_yticklabels(['Our Model'])
    ax2.set_xlabel('')
    ax2.set_title('Our Model Predictions (Red border = mismatch)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    # 3. Reference Probability Heatmap
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    im1 = ax3.imshow(ref_probabilities.T, aspect='auto', cmap='viridis',
                     extent=[0, time_hours[-1], -0.5, MODEL_OUTPUT_SIZE-0.5],
                     origin='lower', interpolation='nearest')
    ax3.set_yticks(range(MODEL_OUTPUT_SIZE))
    ax3.set_yticklabels(SLEEP_STAGE_NAMES)
    ax3.set_ylabel('Sleep Stage')
    ax3.set_xlabel('')
    ax3.set_title('Reference Probability Distribution', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax3, label='Probability')

    # 4. Our Probability Heatmap
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    im2 = ax4.imshow(our_probabilities.T, aspect='auto', cmap='viridis',
                     extent=[0, time_hours[-1], -0.5, MODEL_OUTPUT_SIZE-0.5],
                     origin='lower', interpolation='nearest')
    ax4.set_yticks(range(MODEL_OUTPUT_SIZE))
    ax4.set_yticklabels(SLEEP_STAGE_NAMES)
    ax4.set_ylabel('Sleep Stage')
    ax4.set_xlabel('Time (hours)')
    ax4.set_title('Our Model Probability Distribution', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax4, label='Probability')

    # Add legend for sleep stages
    legend_elements = [mpatches.Patch(facecolor=stage_colors[i], label=SLEEP_STAGE_NAMES[i])
                      for i in range(MODEL_OUTPUT_SIZE)]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    # Add overall statistics
    matches = np.sum(ref_predictions == our_predictions)
    accuracy = 100.0 * matches / num_epochs
    fig.suptitle(f'Sleep Stage Classification Comparison\n'
                f'Accuracy: {matches}/{num_epochs} ({accuracy:.1f}%)',
                fontsize=14, fontweight='bold', y=0.995)

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'sleep_stage_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization: {output_path}")

    # Also save a difference plot
    fig2, ax = plt.subplots(figsize=(16, 6))

    # Plot probability differences
    prob_diff = np.abs(our_probabilities - ref_probabilities)
    im = ax.imshow(prob_diff.T, aspect='auto', cmap='hot',
                   extent=[0, time_hours[-1], -0.5, MODEL_OUTPUT_SIZE-0.5],
                   origin='lower', interpolation='nearest')
    ax.set_yticks(range(MODEL_OUTPUT_SIZE))
    ax.set_yticklabels(SLEEP_STAGE_NAMES)
    ax.set_ylabel('Sleep Stage')
    ax.set_xlabel('Time (hours)')
    ax.set_title('Absolute Probability Differences (|Our - Reference|)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='|Probability Difference|')

    # Mark mismatches
    mismatch_times = time_hours[ref_predictions != our_predictions]
    if len(mismatch_times) > 0:
        for t in mismatch_times:
            ax.axvline(t, color='cyan', alpha=0.3, linewidth=0.5)

    output_path2 = os.path.join(output_dir, 'probability_differences.png')
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved difference plot: {output_path2}")

    plt.close('all')

    return output_path, output_path2


def main():
    print("="*70)
    print("ML Pipeline Testing Script")
    print("="*70)
    print(f"Reference data directory: {DEBUG_DIR}")

    # Check if debug directory exists
    if not DEBUG_DIR.exists():
        print(f"\n✗ ERROR: Debug directory not found: {DEBUG_DIR}")
        print("Please ensure reference files are in data/example_datasets/debug/")
        return

    # Load reference data
    try:
        ref_data = load_reference_data()
    except FileNotFoundError as e:
        print(f"\n✗ ERROR: {e}")
        return

    # Get total number of epochs from reference data
    total_epochs = ref_data['normalized'].shape[0]
    print(f"\nTotal epochs available: {total_epochs} ({total_epochs * 30 / 3600:.2f} hours)")

    # Test normalization on first 10 epochs (quick sanity check)
    norm_results = test_normalization(ref_data, num_epochs=10)

    # Test inference on ALL epochs
    inf_results = test_inference(ref_data, num_epochs=total_epochs)

    if inf_results:
        predictions_matched, mean_diffs, our_predictions, our_probabilities = inf_results

        # Analyze mismatches
        analyze_mismatches(
            ref_data['predictions'][:total_epochs],
            our_predictions,
            ref_data['probabilities'][:total_epochs],
            our_probabilities
        )

        # Create visualizations
        visualize_results(
            ref_data['predictions'][:total_epochs],
            our_predictions,
            ref_data['probabilities'][:total_epochs],
            our_probabilities,
            eeg_data=ref_data['eeg']
        )

        # Final summary
        print_section("FINAL SUMMARY")
        print(f"✓ Reference data loaded successfully")
        print(f"✓ Normalization test complete")
        print(f"✓ Inference test complete: {predictions_matched}/{total_epochs} predictions matched ({100*predictions_matched/total_epochs:.1f}%)")
        print(f"✓ Mismatch analysis complete")
        print(f"✓ Visualizations saved to debug_outputs/")

        print("\nGenerated outputs:")
        print("  - debug_outputs/sleep_stage_comparison.png - Hypnogram and probability heatmaps")
        print("  - debug_outputs/probability_differences.png - Difference heatmap")
        print("  - debug_outputs/confusion_matrix.npy - Confusion matrix")
        print("  - debug_outputs/mismatch_indices.npy - Indices of mismatched epochs")
    else:
        print_section("FINAL SUMMARY")
        print(f"✓ Reference data loaded successfully")
        print(f"✓ Normalization test complete")
        print(f"⚠ Inference test skipped (model or TensorFlow not available)")

    print("\nNext steps:")
    print("1. Review visualizations and mismatch analysis")
    print("2. If baseline accuracy (~80%) is confirmed, proceed to fix firmware")
    print("3. Upload debug_ml_pipeline.cpp to Teensy")
    print("4. Copy debug files to SD card /debug/ folder")
    print("5. Run on-device tests to validate firmware implementation")


if __name__ == '__main__':
    main()
