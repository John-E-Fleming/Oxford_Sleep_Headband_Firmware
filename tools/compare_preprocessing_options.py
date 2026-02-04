#!/usr/bin/env python3
"""
Compare confusion matrices across multiple preprocessing pipeline options.

Usage:
    python compare_preprocessing_options.py                    # Auto-detect all options
    python compare_preprocessing_options.py Default Option_A   # Compare specific options
    python compare_preprocessing_options.py --list             # List available options
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import os
import sys
import argparse

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALIDATION_DIR = os.path.join(BASE_DIR, "data", "validation_testing")
# Look for reference in validation_testing first, then fall back to data/
REFERENCE_PATH_PRIMARY = os.path.join(VALIDATION_DIR, "reference_predictions.csv")
REFERENCE_PATH_FALLBACK = os.path.join(BASE_DIR, "data", "reference_predictions.csv")

# Stage labels in clinical order
STAGE_LABELS = ['Wake', 'REM', 'N1', 'N2', 'N3']

def find_available_options():
    """Find all available preprocessing options in validation_testing directory."""
    options = []
    if os.path.exists(VALIDATION_DIR):
        for name in os.listdir(VALIDATION_DIR):
            option_dir = os.path.join(VALIDATION_DIR, name)
            if os.path.isdir(option_dir):
                # Check if it has a predictions file
                pred_files = [f for f in os.listdir(option_dir) if f.endswith('_predictions.csv')]
                if pred_files:
                    options.append(name)
    return sorted(options)

def load_reference():
    """Load reference predictions from Ali's model."""
    # Check primary location first (validation_testing folder)
    if os.path.exists(REFERENCE_PATH_PRIMARY):
        ref_path = REFERENCE_PATH_PRIMARY
    elif os.path.exists(REFERENCE_PATH_FALLBACK):
        ref_path = REFERENCE_PATH_FALLBACK
    else:
        raise FileNotFoundError(
            f"Reference predictions not found in:\n"
            f"  {REFERENCE_PATH_PRIMARY}\n"
            f"  {REFERENCE_PATH_FALLBACK}"
        )

    print(f"Using reference: {ref_path}")
    df = pd.read_csv(ref_path)
    return df['Predicted_Stage'].values

def load_teensy_predictions(option_name):
    """Load Teensy predictions for a given option."""
    option_dir = os.path.join(VALIDATION_DIR, option_name)

    # Find the predictions file
    pred_files = [f for f in os.listdir(option_dir) if f.endswith('_predictions.csv')]
    if not pred_files:
        raise FileNotFoundError(f"No predictions file found in {option_dir}")

    pred_path = os.path.join(option_dir, pred_files[0])
    df = pd.read_csv(pred_path)
    return df['predicted_stage'].values

def compute_metrics(y_true, y_pred, name):
    """Compute and print metrics."""
    agreement = np.sum(y_true == y_pred)
    total = len(y_true)
    pct = 100 * agreement / total

    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Agreement: {agreement}/{total} ({pct:.1f}%)")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=STAGE_LABELS, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=STAGE_LABELS)
    return cm, pct

def plot_confusion_matrices(cms, titles, agreements, output_path):
    """Plot side-by-side confusion matrices for any number of options."""
    n = len(cms)

    # Determine figure size based on number of matrices
    fig_width = min(5 * n, 20)  # Cap at 20 inches wide
    fig, axes = plt.subplots(1, n, figsize=(fig_width, 5))

    if n == 1:
        axes = [axes]

    for ax, cm, title, agreement in zip(axes, cms, titles, agreements):
        # Normalize by row (true labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)

        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f"{title}\n({agreement:.1f}% agreement)", fontsize=10, fontweight='bold')

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Labels
        ax.set_xticks(np.arange(len(STAGE_LABELS)))
        ax.set_yticks(np.arange(len(STAGE_LABELS)))
        ax.set_xticklabels(STAGE_LABELS)
        ax.set_yticklabels(STAGE_LABELS)
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('Reference (True)', fontsize=9)

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        # Add text annotations
        thresh = 0.5
        for i in range(len(STAGE_LABELS)):
            for j in range(len(STAGE_LABELS)):
                count = cm[i, j]
                pct = cm_normalized[i, j]
                text_color = 'white' if pct > thresh else 'black'
                ax.text(j, i, f'{count}\n({pct:.0%})',
                       ha='center', va='center', color=text_color, fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved confusion matrix comparison to: {output_path}")
    plt.show()

def compare_options(ref, predictions_dict):
    """Compare where different options agree/disagree."""
    option_names = list(predictions_dict.keys())

    if len(option_names) < 2:
        return

    print(f"\n{'='*60}")
    print("Pairwise Comparison Summary")
    print(f"{'='*60}")

    # Compare each pair
    for i, name1 in enumerate(option_names):
        for name2 in option_names[i+1:]:
            pred1 = predictions_dict[name1]
            pred2 = predictions_dict[name2]

            diff_indices = np.where(pred1 != pred2)[0]
            print(f"\n{name1} vs {name2}:")
            print(f"  Epochs where they differ: {len(diff_indices)}/{len(pred1)}")

            if len(diff_indices) > 0:
                # Count which is correct when they differ
                correct_1 = np.sum(pred1[diff_indices] == ref[diff_indices])
                correct_2 = np.sum(pred2[diff_indices] == ref[diff_indices])
                both_wrong = len(diff_indices) - correct_1 - correct_2

                print(f"  When they disagree:")
                print(f"    {name1} correct: {correct_1}")
                print(f"    {name2} correct: {correct_2}")
                print(f"    Both wrong: {both_wrong}")

def print_per_stage_comparison(ref, predictions_dict):
    """Print per-stage recall comparison across all options."""
    print(f"\n{'='*60}")
    print("Per-Stage Recall Comparison")
    print(f"{'='*60}")

    # Header
    header = f"{'Stage':<8}"
    for name in predictions_dict.keys():
        header += f"{name:>12}"
    print(header)
    print("-" * len(header))

    # Per-stage recall
    for stage in STAGE_LABELS:
        row = f"{stage:<8}"
        stage_mask = ref == stage
        stage_count = np.sum(stage_mask)

        for name, pred in predictions_dict.items():
            if stage_count > 0:
                recall = np.sum(pred[stage_mask] == stage) / stage_count
                row += f"{recall*100:>11.1f}%"
            else:
                row += f"{'N/A':>12}"
        print(row)

def main():
    parser = argparse.ArgumentParser(description='Compare preprocessing pipeline confusion matrices')
    parser.add_argument('options', nargs='*', help='Options to compare (default: all available)')
    parser.add_argument('--list', action='store_true', help='List available options')
    parser.add_argument('--output', '-o', help='Output filename (default: auto-generated)')
    args = parser.parse_args()

    # List available options
    available = find_available_options()

    if args.list:
        print("Available preprocessing options:")
        for opt in available:
            print(f"  - {opt}")
        return

    # Determine which options to compare
    if args.options:
        options_to_compare = args.options
        # Validate
        for opt in options_to_compare:
            if opt not in available:
                print(f"Error: Option '{opt}' not found in {VALIDATION_DIR}")
                print(f"Available options: {', '.join(available)}")
                return
    else:
        options_to_compare = available

    if not options_to_compare:
        print(f"No preprocessing options found in {VALIDATION_DIR}")
        print("Run validation tests first to generate prediction files.")
        return

    print(f"Comparing {len(options_to_compare)} preprocessing options: {', '.join(options_to_compare)}")

    # Load reference
    print("\nLoading reference predictions...")
    ref = load_reference()
    print(f"Reference epochs: {len(ref)}")

    # Load predictions for each option
    predictions_dict = {}
    for opt in options_to_compare:
        print(f"Loading {opt} predictions...")
        pred = load_teensy_predictions(opt)
        print(f"  {opt} epochs: {len(pred)}")
        predictions_dict[opt] = pred

    # Align all predictions to same length (both use 0-indexed epochs)
    min_len = min(len(ref), *[len(p) for p in predictions_dict.values()])

    # Direct comparison: reference epoch N matches logged epoch N
    ref_aligned = ref[:min_len]
    aligned_predictions = {name: pred[:min_len] for name, pred in predictions_dict.items()}

    print(f"\nAligned epochs for comparison: {len(ref_aligned)}")

    # Compute metrics and confusion matrices for each option
    cms = []
    titles = []
    agreements = []

    for name, pred in aligned_predictions.items():
        cm, agreement = compute_metrics(ref_aligned, pred, name)
        cms.append(cm)
        titles.append(name)
        agreements.append(agreement)

    # Pairwise comparison
    compare_options(ref_aligned, aligned_predictions)

    # Per-stage comparison
    print_per_stage_comparison(ref_aligned, aligned_predictions)

    # Generate output filename
    if args.output:
        output_path = args.output
    else:
        options_str = "_vs_".join(options_to_compare)
        output_path = os.path.join(BASE_DIR, f"confusion_matrix_comparison_{options_str}.png")

    # Plot confusion matrices
    plot_confusion_matrices(cms, titles, agreements, output_path)

if __name__ == "__main__":
    main()
