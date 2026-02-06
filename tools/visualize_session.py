#!/usr/bin/env python3
"""
Session Visualization Tool
===========================

Creates a combined spectrogram + hypnogram visualization from real-time
session data logged to SD card. Optionally includes accelerometer data.

Input files (from /realtime_logs/ on SD card):
    - predictions_HHMMSS.csv: epoch, timestamp_s, prob_*, predicted_stage
    - eeg_100hz_HHMMSS.csv: sample_index, timestamp_ms, eeg_uv
    - (optional) accel_HHMMSS.csv: sample_index, timestamp_ms, accel_x, accel_y, accel_z

Usage:
    python tools/visualize_session.py <predictions_file> <eeg_file>
    python tools/visualize_session.py predictions.csv eeg_100hz.csv --accel accel.csv

    With accelerometer magnitude display:
    python tools/visualize_session.py predictions.csv eeg_100hz.csv --accel accel.csv --accel-mode magnitude

    Time axis in hours:
    python tools/visualize_session.py predictions.csv eeg_100hz.csv --hours

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


def load_accelerometer(filepath):
    """Load accelerometer CSV file.

    Supports both old format (accel_x, accel_y, accel_z) and
    new format (accel_x_g, accel_y_g, accel_z_g in g units).
    """
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} accelerometer samples from {filepath}")

    # Check for new format (with _g suffix indicating g units)
    if 'accel_x_g' in df.columns:
        accel_x = df['accel_x_g'].values
        accel_y = df['accel_y_g'].values
        accel_z = df['accel_z_g'].values
        units = 'g'
    else:
        # Old format - raw values, convert to g
        ACCEL_SCALE = 16.0 / 4095.0
        accel_x = df['accel_x'].values * ACCEL_SCALE
        accel_y = df['accel_y'].values * ACCEL_SCALE
        accel_z = df['accel_z'].values * ACCEL_SCALE
        units = 'g (converted)'

    print(f"  Accelerometer units: {units}")
    return accel_x, accel_y, accel_z, df['timestamp_ms'].values


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


def plot_session(eeg_data, predictions_df, output_file=None, title="Sleep Session",
                 accel_data=None, accel_mode='xyz', show_hours=False, skip_spectrogram=False,
                 show_probabilities=False):
    """Create combined spectrogram + hypnogram plot with optional accelerometer.

    Args:
        eeg_data: Preprocessed EEG data at 100Hz
        predictions_df: DataFrame with predictions
        output_file: Output file path (optional)
        title: Plot title
        accel_data: Tuple of (accel_x, accel_y, accel_z, timestamps_ms) or None
        accel_mode: 'xyz' for individual traces, 'magnitude' for sqrt(x^2+y^2+z^2)
        show_hours: If True, show time axis in hours instead of seconds
        skip_spectrogram: If True, skip spectrogram computation (faster)
        show_probabilities: If True, show probability panel instead of spectrogram
    """
    import matplotlib.gridspec as gridspec

    # If showing probabilities, skip spectrogram
    if show_probabilities:
        skip_spectrogram = True

    # Compute spectrogram if needed
    if not skip_spectrogram:
        f, t_spec, Sxx = compute_spectrogram(eeg_data)

    # Get hypnogram data
    t_hyp, stages = create_hypnogram(predictions_df)

    # Determine number of rows
    has_accel = accel_data is not None
    has_spec = not skip_spectrogram
    has_probs = show_probabilities

    if has_accel and (has_spec or has_probs):
        n_rows = 3
        height_ratios = [1, 2.5, 1.5]
    elif has_accel:
        n_rows = 2
        height_ratios = [1, 1.5]
    elif has_spec or has_probs:
        n_rows = 2
        height_ratios = [1, 2.5]
    else:
        n_rows = 1
        height_ratios = [1]

    # Create figure with GridSpec
    fig = plt.figure(figsize=(14, 3 + 3 * n_rows))
    gs = gridspec.GridSpec(n_rows, 2, height_ratios=height_ratios, width_ratios=[50, 1],
                           hspace=0.08, wspace=0.02)

    # Time conversion factor
    time_scale = 3600 if show_hours else 1
    time_label = 'Time (hours)' if show_hours else 'Time (seconds)'

    # Scale time data
    t_hyp_scaled = t_hyp / time_scale

    # --- Hypnogram (top) ---
    ax_hyp = fig.add_subplot(gs[0, 0])

    # Plot as step function with colored bars
    for i in range(len(stages)):
        start = (t_hyp[i] - EPOCH_DURATION) / time_scale
        end = t_hyp[i] / time_scale
        stage = stages[i]
        ax_hyp.axvspan(start, end, color=STAGE_COLORS[stage], alpha=0.7)

    # Plot step line
    x_steps = []
    y_steps = []
    for i in range(len(stages)):
        start = (t_hyp[i] - EPOCH_DURATION) / time_scale
        end = t_hyp[i] / time_scale
        x_steps.extend([start, end])
        y_steps.extend([stages[i], stages[i]])

    ax_hyp.plot(x_steps, y_steps, 'k-', linewidth=1.5)

    ax_hyp.set_ylabel('Sleep Stage')
    ax_hyp.set_yticks([0, 1, 2, 3, 4])
    ax_hyp.set_yticklabels(STAGE_NAMES)
    ax_hyp.set_ylim(-0.5, 4.5)
    ax_hyp.invert_yaxis()
    ax_hyp.set_title(title)
    ax_hyp.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=STAGE_COLORS[i], alpha=0.7, label=STAGE_NAMES[i])
                       for i in range(5)]
    ax_hyp.legend(handles=legend_elements, loc='upper right', ncol=5, fontsize=8)

    max_time = t_hyp[-1] / time_scale
    current_row = 1

    # --- Spectrogram ---
    if has_spec:
        ax_spec = fig.add_subplot(gs[current_row, 0], sharex=ax_hyp)
        cax = fig.add_subplot(gs[current_row, 1])

        t_spec_scaled = t_spec / time_scale

        # Focus on sleep-relevant frequencies
        freq_mask = (f >= 0.5) & (f <= 30)
        f_plot = f[freq_mask]
        Sxx_plot = Sxx[freq_mask, :]

        im = ax_spec.pcolormesh(t_spec_scaled, f_plot, Sxx_plot,
                                shading='gouraud', cmap='viridis',
                                vmin=np.percentile(Sxx_plot, 5),
                                vmax=np.percentile(Sxx_plot, 95))

        ax_spec.set_ylabel('Frequency (Hz)')
        ax_spec.set_ylim(0.5, 30)

        # Frequency band lines
        ax_spec.axhline(y=4, color='white', linestyle='--', alpha=0.5, linewidth=0.5)
        ax_spec.axhline(y=8, color='white', linestyle='--', alpha=0.5, linewidth=0.5)
        ax_spec.axhline(y=12, color='white', linestyle='--', alpha=0.5, linewidth=0.5)

        fig.colorbar(im, cax=cax, label='Power (dB)')

        max_time = max(max_time, t_spec_scaled[-1])
        current_row += 1

        if not has_accel:
            ax_spec.set_xlabel(time_label)

    # --- Probability Heatmap Panel ---
    if has_probs:
        ax_prob = fig.add_subplot(gs[current_row, 0], sharex=ax_hyp)
        cax_prob = fig.add_subplot(gs[current_row, 1])

        # Get probability columns in order: Wake, REM, N1, N2, N3 (matching hypnogram order)
        prob_cols = ['prob_wake', 'prob_rem', 'prob_n1', 'prob_n2', 'prob_n3']
        prob_labels = ['Wake', 'REM', 'N1 VeryLight', 'N2 Light', 'N3 Deep']

        # Build probability matrix (stages x epochs)
        n_epochs = len(predictions_df)
        probs = np.zeros((5, n_epochs))
        for i, col in enumerate(prob_cols):
            probs[i, :] = predictions_df[col].values

        # Create time edges for pcolormesh (need n+1 edges for n cells)
        timestamps = predictions_df['timestamp_s'].values
        # Each epoch covers [t - EPOCH_DURATION, t], so create edges accordingly
        time_edges = np.zeros(n_epochs + 1)
        time_edges[0] = (timestamps[0] - EPOCH_DURATION) / time_scale
        time_edges[1:] = timestamps / time_scale

        # Stage edges (0.5 spacing centered on 0, 1, 2, 3, 4)
        stage_edges = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])

        # Plot heatmap
        im = ax_prob.pcolormesh(time_edges, stage_edges, probs,
                                shading='flat', cmap='viridis',
                                vmin=0, vmax=1)

        ax_prob.set_ylabel('Sleep Stage')
        ax_prob.set_yticks([0, 1, 2, 3, 4])
        ax_prob.set_yticklabels(prob_labels)
        ax_prob.set_ylim(-0.5, 4.5)
        ax_prob.invert_yaxis()  # Wake at top, N3 at bottom

        fig.colorbar(im, cax=cax_prob, label='Probability')

        max_time = max(max_time, time_edges[-1])
        current_row += 1

        if not has_accel:
            ax_prob.set_xlabel(time_label)

    # --- Accelerometer ---
    if has_accel:
        accel_x, accel_y, accel_z, accel_t_ms = accel_data
        accel_t = accel_t_ms / 1000.0 / time_scale  # Convert ms to time_scale units

        ax_accel = fig.add_subplot(gs[current_row, 0], sharex=ax_hyp)

        if accel_mode == 'magnitude':
            # Plot activity magnitude
            magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
            ax_accel.plot(accel_t, magnitude, 'b-', linewidth=0.5, alpha=0.8)
            ax_accel.set_ylabel('Activity (g)')
        else:
            # Plot individual X, Y, Z traces
            ax_accel.plot(accel_t, accel_x, 'r-', linewidth=0.5, alpha=0.7, label='X')
            ax_accel.plot(accel_t, accel_y, 'g-', linewidth=0.5, alpha=0.7, label='Y')
            ax_accel.plot(accel_t, accel_z, 'b-', linewidth=0.5, alpha=0.7, label='Z')
            ax_accel.legend(loc='upper right', ncol=3, fontsize=8)
            ax_accel.set_ylabel('Acceleration (g)')

        ax_accel.set_xlabel(time_label)
        ax_accel.grid(True, alpha=0.3)

        max_time = max(max_time, accel_t[-1])

    # Set x-axis limits
    ax_hyp.set_xlim(0, max_time)

    plt.tight_layout()

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

    # Accelerometer options
    parser.add_argument('--accel', help='Accelerometer CSV file')
    parser.add_argument('--accel-mode', choices=['xyz', 'magnitude'], default='xyz',
                        help='Accelerometer display mode: xyz (individual traces) or magnitude')

    # Display options
    parser.add_argument('--hours', action='store_true',
                        help='Show time axis in hours instead of seconds')
    parser.add_argument('--no-spec', action='store_true',
                        help='Skip spectrogram computation (faster)')
    parser.add_argument('--show-probs', action='store_true',
                        help='Show probability panel instead of spectrogram')

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

    # Try to load EEG data (may be empty or file may not exist)
    eeg_data = np.array([])
    if eeg_file.exists():
        eeg_data = load_eeg(eeg_file)
    else:
        print(f"WARNING: EEG file not found: {eeg_file}")

    # Load accelerometer if provided
    accel_data = None
    if args.accel:
        accel_file = Path(args.accel)
        if accel_file.exists():
            accel_data = load_accelerometer(accel_file)
        else:
            print(f"WARNING: Accelerometer file not found: {accel_file}")

    # Print summary
    print_summary(predictions_df)

    # Generate output filename if not specified
    output_file = args.output
    if not output_file:
        output_file = pred_file.stem.replace('predictions', 'session_plot') + '.png'

    # Auto-skip spectrogram if EEG data is empty or too short
    skip_spectrogram = args.no_spec
    if len(eeg_data) < 256:  # Need at least nperseg samples for spectrogram
        if not args.no_spec:
            print("NOTE: EEG data too short for spectrogram, skipping (use --no-spec to suppress this)")
        skip_spectrogram = True

    # Create main visualization
    print("Generating visualization...")
    plot_session(eeg_data, predictions_df, output_file,
                 title=f"Sleep Session: {pred_file.stem}",
                 accel_data=accel_data,
                 accel_mode=args.accel_mode,
                 show_hours=args.hours,
                 skip_spectrogram=skip_spectrogram,
                 show_probabilities=args.show_probs)

    # Optionally plot probabilities
    if args.probs:
        prob_output = output_file.replace('.png', '_probs.png')
        plot_probability_timeline(predictions_df, prob_output)


if __name__ == '__main__':
    main()
