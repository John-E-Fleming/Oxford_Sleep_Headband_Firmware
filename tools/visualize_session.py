#!/usr/bin/env python3
"""
Session Visualization Tool
===========================

Creates a combined spectrogram + hypnogram visualization from real-time
session data logged to SD card.

Input files (from /realtime_logs/ on SD card):
    - predictions_HHMMSS.csv: epoch, timestamp_s, prob_*, predicted_stage
    - eeg_100hz_HHMMSS.csv: sample_index, timestamp_ms, eeg_uv

Usage:
    python tools/visualize_session.py <predictions_file> <eeg_file>
    python tools/visualize_session.py predictions_120530.csv eeg_100hz_120530.csv

    Or with a directory containing paired files:
    python tools/visualize_session.py --dir /path/to/realtime_logs/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import signal
from pathlib import Path
import argparse
import sys

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# EEG parameters
SAMPLE_RATE = 100  # Hz (logged at 100Hz after preprocessing)
EPOCH_DURATION = 30  # seconds

# Sleep stage mapping (clinical hypnogram order: Wake at top, N3 at bottom)
STAGE_MAP = {'Wake': 0, 'REM': 1, 'N1': 2, 'N2': 3, 'N3': 4}
STAGE_COLORS = ['#E74C3C', '#9B59B6', '#F39C12', '#3498DB', '#2ECC71']  # Wake, REM, N1, N2, N3
STAGE_NAMES = ['Wake', 'REM', 'N1', 'N2', 'N3']


def load_predictions(filepath):
    """Load predictions CSV file."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} epochs from {filepath}")
    return df


def load_eeg(filepath):
    """Load EEG CSV file."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} samples from {filepath}")
    return df['eeg_uv'].values


def compute_spectrogram(eeg_data, fs=SAMPLE_RATE, nperseg=256, noverlap=128):
    """Compute spectrogram of EEG data."""
    f, t, Sxx = signal.spectrogram(
        eeg_data,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density'
    )
    # Convert to dB scale
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    return f, t, Sxx_db


def create_hypnogram(predictions_df):
    """Create hypnogram array from predictions dataframe."""
    stages = predictions_df['predicted_stage'].map(STAGE_MAP).values
    timestamps = predictions_df['timestamp_s'].values
    return timestamps, stages


def plot_session(eeg_data, predictions_df, output_file=None, title="Sleep Session"):
    """Create combined spectrogram + hypnogram plot."""
    import matplotlib.gridspec as gridspec

    # Compute spectrogram
    f, t_spec, Sxx = compute_spectrogram(eeg_data)

    # Get hypnogram data
    t_hyp, stages = create_hypnogram(predictions_df)

    # Create figure with GridSpec: 2 rows, 2 cols (main plots + colorbar column)
    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[50, 1],
                           hspace=0.08, wspace=0.02)

    ax_hyp = fig.add_subplot(gs[0, 0])
    ax_spec = fig.add_subplot(gs[1, 0], sharex=ax_hyp)
    cax = fig.add_subplot(gs[1, 1])  # Colorbar axes aligned with spectrogram only

    # --- Hypnogram (top) ---

    # Plot as step function with colored bars
    for i in range(len(stages)):
        start = t_hyp[i] - EPOCH_DURATION
        end = t_hyp[i]
        stage = stages[i]
        ax_hyp.axvspan(start, end, color=STAGE_COLORS[stage], alpha=0.7)

    # Plot step line
    # Create step-like x coordinates
    x_steps = []
    y_steps = []
    for i in range(len(stages)):
        start = t_hyp[i] - EPOCH_DURATION
        end = t_hyp[i]
        x_steps.extend([start, end])
        y_steps.extend([stages[i], stages[i]])

    ax_hyp.plot(x_steps, y_steps, 'k-', linewidth=1.5)

    ax_hyp.set_ylabel('Sleep Stage')
    ax_hyp.set_yticks([0, 1, 2, 3, 4])
    ax_hyp.set_yticklabels(STAGE_NAMES)
    ax_hyp.set_ylim(-0.5, 4.5)
    ax_hyp.invert_yaxis()  # Wake at top, deep sleep at bottom
    ax_hyp.set_title(title)
    ax_hyp.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=STAGE_COLORS[i], alpha=0.7, label=STAGE_NAMES[i])
                       for i in range(5)]
    ax_hyp.legend(handles=legend_elements, loc='upper right', ncol=5, fontsize=8)

    # --- Spectrogram (bottom) ---

    # Focus on sleep-relevant frequencies (0.5-30 Hz)
    freq_mask = (f >= 0.5) & (f <= 30)
    f_plot = f[freq_mask]
    Sxx_plot = Sxx[freq_mask, :]

    # Plot spectrogram
    im = ax_spec.pcolormesh(t_spec, f_plot, Sxx_plot,
                            shading='gouraud', cmap='viridis',
                            vmin=np.percentile(Sxx_plot, 5),
                            vmax=np.percentile(Sxx_plot, 95))

    ax_spec.set_ylabel('Frequency (Hz)')
    ax_spec.set_xlabel('Time (seconds)')
    ax_spec.set_ylim(0.5, 30)

    # Set x-axis limits to remove blank space (use max timestamp from data)
    max_time = max(t_hyp[-1], t_spec[-1])
    ax_hyp.set_xlim(0, max_time)
    ax_spec.set_xlim(0, max_time)

    # Add frequency band annotations (horizontal lines only)
    ax_spec.axhline(y=4, color='white', linestyle='--', alpha=0.5, linewidth=0.5)
    ax_spec.axhline(y=8, color='white', linestyle='--', alpha=0.5, linewidth=0.5)
    ax_spec.axhline(y=12, color='white', linestyle='--', alpha=0.5, linewidth=0.5)

    # Add colorbar in dedicated axes (already aligned with spectrogram via GridSpec)
    cbar = fig.colorbar(im, cax=cax, label='Power (dB)')

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {output_file}")

    plt.show()
    return fig


def plot_probability_timeline(predictions_df, output_file=None):
    """Plot probability timelines for each sleep stage."""

    fig, ax = plt.subplots(figsize=(14, 5))

    timestamps = predictions_df['timestamp_s'].values

    # Plot each stage probability
    prob_cols = ['prob_wake', 'prob_n1', 'prob_n2', 'prob_n3', 'prob_rem']
    for i, col in enumerate(prob_cols):
        ax.plot(timestamps, predictions_df[col].values,
                color=STAGE_COLORS[i], label=STAGE_NAMES[i],
                linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)
    ax.set_title('Sleep Stage Probabilities Over Time')
    ax.legend(loc='upper right', ncol=5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved probability plot to {output_file}")

    plt.show()
    return fig


def print_summary(predictions_df):
    """Print session summary statistics."""
    total_epochs = len(predictions_df)
    total_time_min = total_epochs * EPOCH_DURATION / 60

    print("\n" + "="*50)
    print("SESSION SUMMARY")
    print("="*50)
    print(f"Total epochs: {total_epochs}")
    print(f"Total duration: {total_time_min:.1f} minutes ({total_time_min/60:.2f} hours)")

    print("\nStage distribution:")
    stage_counts = predictions_df['predicted_stage'].value_counts()
    for stage in STAGE_NAMES:
        count = stage_counts.get(stage, 0)
        pct = 100 * count / total_epochs
        time_min = count * EPOCH_DURATION / 60
        print(f"  {stage:5s}: {count:4d} epochs ({pct:5.1f}%) = {time_min:.1f} min")

    print("="*50 + "\n")


def find_paired_files(directory):
    """Find paired prediction and EEG files in a directory."""
    directory = Path(directory)
    pred_files = sorted(directory.glob('predictions_*.csv'))

    pairs = []
    for pred_file in pred_files:
        # Extract timestamp from filename
        timestamp = pred_file.stem.replace('predictions_', '')
        eeg_file = directory / f'eeg_100hz_{timestamp}.csv'

        if eeg_file.exists():
            pairs.append((pred_file, eeg_file))
            print(f"Found pair: {pred_file.name} + {eeg_file.name}")
        else:
            print(f"Warning: No matching EEG file for {pred_file.name}")

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description='Visualize sleep session from SD card logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('predictions', nargs='?', help='Predictions CSV file')
    parser.add_argument('eeg', nargs='?', help='EEG CSV file')
    parser.add_argument('--dir', '-d', help='Directory containing paired log files')
    parser.add_argument('--output', '-o', help='Output image file (optional)')
    parser.add_argument('--probs', action='store_true',
                        help='Also plot probability timeline')

    args = parser.parse_args()

    # Determine input files
    if args.dir:
        pairs = find_paired_files(args.dir)
        if not pairs:
            print("No paired files found in directory")
            return
        # Use the most recent pair
        pred_file, eeg_file = pairs[-1]
        print(f"\nUsing most recent pair: {pred_file.name}")
    elif args.predictions and args.eeg:
        pred_file = Path(args.predictions)
        eeg_file = Path(args.eeg)
    else:
        parser.print_help()
        print("\nError: Provide either --dir or both predictions and eeg files")
        return

    # Load data
    print("\nLoading data...")
    predictions_df = load_predictions(pred_file)
    eeg_data = load_eeg(eeg_file)

    # Print summary
    print_summary(predictions_df)

    # Generate output filename if not specified
    output_file = args.output
    if not output_file:
        output_file = pred_file.stem.replace('predictions', 'session_plot') + '.png'

    # Create main visualization
    print("Generating spectrogram + hypnogram plot...")
    plot_session(eeg_data, predictions_df, output_file,
                 title=f"Sleep Session: {pred_file.stem}")

    # Optionally plot probabilities
    if args.probs:
        prob_output = output_file.replace('.png', '_probs.png')
        plot_probability_timeline(predictions_df, prob_output)


if __name__ == '__main__':
    main()
