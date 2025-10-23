#!/usr/bin/env python3
"""
Read and analyze EEG data logged by the Sleep Headband firmware.

This script reads the binary data files produced by DataLogger and provides
utilities for visualization, comparison, and spectrogram generation.

Usage:
    python read_eeg_data.py <session_name>

Example:
    python read_eeg_data.py batch_1234567890

Files expected:
    - batch_1234567890_raw.bin (raw filtered samples)
    - batch_1234567890_normalized.bin (normalized windows)
    - batch_1234567890_metadata.txt (metadata)
"""

import sys
import numpy as np
import struct
from pathlib import Path
from typing import Tuple, List, Dict


def read_metadata(metadata_file: Path) -> Dict[str, str]:
    """Read metadata from text file."""
    metadata = {}
    if not metadata_file.exists():
        print(f"Warning: Metadata file not found: {metadata_file}")
        return metadata

    with open(metadata_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip()

    return metadata


def read_raw_data(raw_file: Path) -> np.ndarray:
    """Read raw filtered EEG samples from binary file.

    Format: Sequential float32 samples (4 bytes each)

    Returns:
        numpy array of float32 samples
    """
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_file}")

    # Read entire file as float32 array
    data = np.fromfile(raw_file, dtype=np.float32)

    print(f"Loaded {len(data)} raw samples")
    print(f"  Duration: {len(data) / 100.0:.2f} seconds (assuming 100Hz)")
    print(f"  Min: {data.min():.3f}, Max: {data.max():.3f}")
    print(f"  Mean: {data.mean():.3f}, Std: {data.std():.3f}")

    return data


def read_normalized_windows(normalized_file: Path) -> List[Tuple[int, np.ndarray]]:
    """Read normalized windows from binary file.

    Format for each window:
        - epoch_index: int32 (4 bytes)
        - window_size: int32 (4 bytes)
        - samples: float32 array (window_size * 4 bytes)

    Returns:
        List of tuples (epoch_index, window_data)
    """
    if not normalized_file.exists():
        raise FileNotFoundError(f"Normalized data file not found: {normalized_file}")

    windows = []

    with open(normalized_file, 'rb') as f:
        window_num = 0
        while True:
            # Read epoch index
            epoch_bytes = f.read(4)
            if not epoch_bytes or len(epoch_bytes) < 4:
                break

            epoch_index = struct.unpack('i', epoch_bytes)[0]

            # Read window size
            size_bytes = f.read(4)
            if not size_bytes or len(size_bytes) < 4:
                print(f"Warning: Incomplete window at index {window_num}")
                break

            window_size = struct.unpack('i', size_bytes)[0]

            # Read window samples
            samples_bytes = f.read(window_size * 4)
            if len(samples_bytes) < window_size * 4:
                print(f"Warning: Incomplete window data at index {window_num}")
                break

            samples = np.frombuffer(samples_bytes, dtype=np.float32)

            windows.append((epoch_index, samples))
            window_num += 1

    print(f"Loaded {len(windows)} normalized windows")
    if windows:
        print(f"  Window size: {len(windows[0][1])} samples")
        print(f"  First epoch: {windows[0][0]}, Last epoch: {windows[-1][0]}")

        # Check normalization
        all_samples = np.concatenate([w[1] for w in windows])
        print(f"  Overall mean: {all_samples.mean():.6f} (should be ~0)")
        print(f"  Overall std: {all_samples.std():.6f} (should be ~1)")

    return windows


def plot_raw_data(raw_data: np.ndarray, sample_rate: float = 100.0,
                  duration_seconds: float = 30.0):
    """Plot a segment of raw data."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    num_samples = int(duration_seconds * sample_rate)
    num_samples = min(num_samples, len(raw_data))

    time_axis = np.arange(num_samples) / sample_rate

    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, raw_data[:num_samples], linewidth=0.5)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title(f'Raw Filtered EEG (first {duration_seconds}s)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('raw_eeg.png', dpi=150)
    print("Saved raw_eeg.png")


def plot_normalized_window(window_data: np.ndarray, epoch_index: int,
                           sample_rate: float = 100.0):
    """Plot a normalized window."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    time_axis = np.arange(len(window_data)) / sample_rate

    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, window_data, linewidth=0.5)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Normalized Amplitude (z-score)')
    plt.title(f'Normalized Window (Epoch {epoch_index})')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f'normalized_window_epoch{epoch_index}.png', dpi=150)
    print(f"Saved normalized_window_epoch{epoch_index}.png")


def generate_spectrogram(data: np.ndarray, sample_rate: float = 100.0,
                        window_seconds: float = 4.0):
    """Generate and plot spectrogram."""
    try:
        import matplotlib.pyplot as plt
        from scipy import signal
    except ImportError:
        print("matplotlib and scipy required for spectrogram")
        return

    # Calculate spectrogram
    nperseg = int(window_seconds * sample_rate)
    frequencies, times, Sxx = signal.spectrogram(data, fs=sample_rate,
                                                   nperseg=nperseg,
                                                   noverlap=nperseg//2)

    # Plot spectrogram
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx),
                   shading='gouraud', cmap='viridis')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (seconds)')
    plt.title('EEG Spectrogram')
    plt.colorbar(label='Power (dB)')
    plt.ylim([0, 35])  # Focus on 0-35 Hz range (delta to beta waves)
    plt.tight_layout()
    plt.savefig('spectrogram.png', dpi=150)
    print("Saved spectrogram.png")


def compare_windows(windows: List[Tuple[int, np.ndarray]],
                   indices: List[int] = [0, 1, 2]):
    """Compare multiple normalized windows."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    fig, axes = plt.subplots(len(indices), 1, figsize=(12, 3*len(indices)))
    if len(indices) == 1:
        axes = [axes]

    sample_rate = 100.0

    for ax, idx in zip(axes, indices):
        if idx >= len(windows):
            continue

        epoch, data = windows[idx]
        time_axis = np.arange(len(data)) / sample_rate

        ax.plot(time_axis, data, linewidth=0.5)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('z-score')
        ax.set_title(f'Epoch {epoch} (Window {idx})')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=0.5)

    plt.tight_layout()
    plt.savefig('window_comparison.png', dpi=150)
    print("Saved window_comparison.png")


def save_to_csv(raw_data: np.ndarray, output_file: str = "raw_data.csv"):
    """Save raw data to CSV for external analysis."""
    np.savetxt(output_file, raw_data, delimiter=',',
               header='sample_value', comments='')
    print(f"Saved raw data to {output_file}")


def save_windows_to_csv(windows: List[Tuple[int, np.ndarray]],
                       output_file: str = "normalized_windows.csv"):
    """Save first few normalized windows to CSV."""
    with open(output_file, 'w') as f:
        for epoch, data in windows[:5]:  # Save first 5 windows
            f.write(f"# Epoch {epoch}\n")
            np.savetxt(f, data, delimiter=',')
            f.write("\n")
    print(f"Saved first 5 normalized windows to {output_file}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    session_name = sys.argv[1]

    # Construct file paths
    base_path = Path(".")
    metadata_file = base_path / f"{session_name}_metadata.txt"
    raw_file = base_path / f"{session_name}_raw.bin"
    normalized_file = base_path / f"{session_name}_normalized.bin"

    print(f"Reading data for session: {session_name}")
    print("=" * 60)

    # Read metadata
    print("\n1. Reading metadata...")
    metadata = read_metadata(metadata_file)
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    # Read raw data
    print("\n2. Reading raw data...")
    raw_data = read_raw_data(raw_file)

    # Read normalized windows
    print("\n3. Reading normalized windows...")
    windows = read_normalized_windows(normalized_file)

    # Generate visualizations
    print("\n4. Generating visualizations...")
    plot_raw_data(raw_data, duration_seconds=30.0)

    if windows:
        plot_normalized_window(windows[0][1], windows[0][0])
        if len(windows) >= 3:
            compare_windows(windows, indices=[0, 1, 2])

    # Generate spectrogram from first 5 minutes of raw data
    max_samples = min(len(raw_data), 30000)  # 5 minutes at 100Hz
    generate_spectrogram(raw_data[:max_samples])

    # Save to CSV for comparison
    print("\n5. Saving data to CSV...")
    save_to_csv(raw_data[:3000], "raw_data_first_30s.csv")
    if windows:
        save_windows_to_csv(windows)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("\nGenerated files:")
    print("  - raw_eeg.png (time series plot)")
    print("  - normalized_window_epoch*.png (normalized window)")
    print("  - spectrogram.png (frequency analysis)")
    print("  - window_comparison.png (multiple windows)")
    print("  - raw_data_first_30s.csv (CSV export)")
    print("  - normalized_windows.csv (first 5 windows)")


if __name__ == "__main__":
    main()
