#!/usr/bin/env python3
"""
Phase 2: Teensy vs Python Comparison Tool
==========================================

This script compares checkpoint outputs between Teensy and Python at each
stage of the preprocessing and inference pipeline.

The Teensy outputs checkpoint statistics to serial (not full arrays to avoid slowdown).
This script parses that output and compares against Python values.

Usage:
    1. Run Teensy with debug output enabled
    2. Copy serial output to a file: teensy_debug.txt
    3. Run: python tools/compare_teensy_python.py teensy_debug.txt

Expected checkpoint format from Teensy:
    [CHECKPOINT A] 100Hz preprocessed - Epoch X
    mean=X.XXXX std=X.XXXX min=X.XXXX max=X.XXXX
    first_10: X.XX X.XX ...
    last_10: X.XX X.XX ...

    [CHECKPOINT B] Epoch extraction - Epoch X
    start_sample=XXXXXX end_sample=XXXXXX

    [CHECKPOINT C] Normalization - Epoch X
    mean=X.XXXX std=X.XXXX
    first_10: X.XXXXXX X.XXXXXX ...

    [CHECKPOINT D] Quantization - Epoch X
    scale=X.XXXXXXXXXX zero_point=X
    first_10: X X X ...
    epoch_quantized=X
"""

import numpy as np
from scipy.signal import butter, lfilter, resample
from pathlib import Path
import sys
import re

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Configuration
REFERENCE_EEG = Path('data/example_datasets/debug/1_bandpassed_eeg_single_channel.npy')
RAW_DATA_FILE = Path('data/example_datasets/eeg/SdioLogger_miklos_night_2.bin')

SAMPLE_RATE_ORIGINAL = 4000
SAMPLE_RATE_FILTER = 250
SAMPLE_RATE_TARGET = 100
NUM_CHANNELS = 9
BIPOLAR_POS = 0
BIPOLAR_NEG = 6
EPOCH_SAMPLES = 3000

LOWCUT = 0.5
HIGHCUT = 30
FILTER_ORDER = 5


def print_section(title):
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}")


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def preprocess_raw_data():
    """Preprocess raw 4kHz data to 100Hz (training script method)"""
    print_section("Loading and Preprocessing Raw Data")

    raw_data = np.fromfile(RAW_DATA_FILE, dtype=np.int32)
    num_samples = len(raw_data) // NUM_CHANNELS
    data_multi = raw_data[:num_samples * NUM_CHANNELS].reshape(-1, NUM_CHANNELS)
    bipolar_4khz = (data_multi[:, BIPOLAR_POS] - data_multi[:, BIPOLAR_NEG]).astype(np.float64)

    # Downsample 4kHz -> 250Hz
    num_250 = len(bipolar_4khz) // 16
    data_250hz = np.zeros(num_250, dtype=np.float64)
    for i in range(num_250):
        data_250hz[i] = np.mean(bipolar_4khz[i * 16:(i + 1) * 16])

    # Bandpass filter at 250Hz
    b, a = butter_bandpass(LOWCUT, HIGHCUT, SAMPLE_RATE_FILTER, FILTER_ORDER)
    filtered_250hz = lfilter(b, a, data_250hz)

    # Resample 250Hz -> 100Hz
    num_100 = int(len(filtered_250hz) * SAMPLE_RATE_TARGET / SAMPLE_RATE_FILTER)
    data_100hz = resample(filtered_250hz, num_100).astype(np.float32)

    print(f"Raw data shape: {bipolar_4khz.shape}")
    print(f"250Hz data shape: {data_250hz.shape}")
    print(f"100Hz data shape: {data_100hz.shape}")

    return data_100hz


def generate_python_checkpoints(data_100hz, num_epochs=10):
    """Generate Python checkpoint values for comparison (FLOAT32 model - no quantization)"""

    checkpoints = []

    for epoch_idx in range(num_epochs):
        start = epoch_idx * EPOCH_SAMPLES
        end = start + EPOCH_SAMPLES

        if end > len(data_100hz):
            break

        epoch_data = data_100hz[start:end]

        checkpoint = {
            'epoch': epoch_idx,
            # Checkpoint A: 100Hz preprocessed
            'checkpoint_a': {
                'mean': np.mean(epoch_data),
                'std': np.std(epoch_data),
                'min': np.min(epoch_data),
                'max': np.max(epoch_data),
                'first_10': epoch_data[:10].tolist(),
                'last_10': epoch_data[-10:].tolist(),
            },
            # Checkpoint B: Epoch extraction
            'checkpoint_b': {
                'start_sample': start,
                'end_sample': end,
            },
        }

        # Checkpoint C: Normalization
        mean = np.mean(epoch_data)
        std = np.std(epoch_data)
        normalized = (epoch_data - mean) / std

        checkpoint['checkpoint_c'] = {
            'mean': mean,
            'std': std,
            'norm_first_10': normalized[:10].tolist(),
            'norm_last_10': normalized[-10:].tolist(),
        }

        # Checkpoint D: Model input (FLOAT32 - no quantization needed)
        # The model uses FLOAT32 inputs, not INT8 quantized
        epoch_scaled = epoch_idx / 1000.0

        checkpoint['checkpoint_d'] = {
            'input_type': 'FLOAT32',
            'float_first_10': normalized[:10].tolist(),
            'epoch_index': epoch_idx,
            'epoch_scaled': epoch_scaled,
        }

        checkpoints.append(checkpoint)

    return checkpoints


def parse_teensy_output(filename):
    """Parse Teensy serial output to extract checkpoint values"""

    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    checkpoints = []
    current_checkpoint = None
    current_section = None

    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for checkpoint markers
        if '[CHECKPOINT A]' in line:
            # Extract epoch number
            match = re.search(r'Epoch\s*(\d+)', line)
            if match:
                epoch = int(match.group(1))
                if current_checkpoint and current_checkpoint['epoch'] != epoch:
                    checkpoints.append(current_checkpoint)
                current_checkpoint = {'epoch': epoch, 'checkpoint_a': {}}
                current_section = 'a'

        elif '[CHECKPOINT B]' in line:
            current_section = 'b'
            if current_checkpoint:
                current_checkpoint['checkpoint_b'] = {}

        elif '[CHECKPOINT C]' in line:
            current_section = 'c'
            if current_checkpoint:
                current_checkpoint['checkpoint_c'] = {}

        elif '[CHECKPOINT D]' in line:
            current_section = 'd'
            if current_checkpoint:
                current_checkpoint['checkpoint_d'] = {}

        elif current_checkpoint:
            # Parse key=value pairs
            if 'mean=' in line:
                match = re.search(r'mean=([+-]?\d+\.?\d*)', line)
                if match and current_section in ['a', 'c']:
                    current_checkpoint[f'checkpoint_{current_section}']['mean'] = float(match.group(1))
                match = re.search(r'std=([+-]?\d+\.?\d*)', line)
                if match:
                    current_checkpoint[f'checkpoint_{current_section}']['std'] = float(match.group(1))
                match = re.search(r'min=([+-]?\d+\.?\d*)', line)
                if match:
                    current_checkpoint[f'checkpoint_{current_section}']['min'] = float(match.group(1))
                match = re.search(r'max=([+-]?\d+\.?\d*)', line)
                if match:
                    current_checkpoint[f'checkpoint_{current_section}']['max'] = float(match.group(1))

            elif 'first_10:' in line:
                values = re.findall(r'([+-]?\d+\.?\d*)', line.split('first_10:')[1])
                if values:
                    if current_section == 'a':
                        current_checkpoint['checkpoint_a']['first_10'] = [float(v) for v in values]
                    elif current_section == 'c':
                        current_checkpoint['checkpoint_c']['norm_first_10'] = [float(v) for v in values]
                    elif current_section == 'd':
                        current_checkpoint['checkpoint_d']['quant_first_10'] = [int(v) for v in values]

            elif 'last_10:' in line:
                values = re.findall(r'([+-]?\d+\.?\d*)', line.split('last_10:')[1])
                if values:
                    if current_section == 'a':
                        current_checkpoint['checkpoint_a']['last_10'] = [float(v) for v in values]
                    elif current_section == 'c':
                        current_checkpoint['checkpoint_c']['norm_last_10'] = [float(v) for v in values]

            elif 'scale=' in line:
                match = re.search(r'scale=([+-]?\d+\.?\d*e?[+-]?\d*)', line)
                if match:
                    current_checkpoint['checkpoint_d']['scale'] = float(match.group(1))
                match = re.search(r'zero_point=([+-]?\d+)', line)
                if match:
                    current_checkpoint['checkpoint_d']['zero_point'] = int(match.group(1))

            elif 'epoch_quantized=' in line:
                match = re.search(r'epoch_quantized=([+-]?\d+)', line)
                if match:
                    current_checkpoint['checkpoint_d']['epoch_quantized'] = int(match.group(1))

            elif 'start_sample=' in line:
                match = re.search(r'start_sample=(\d+)', line)
                if match:
                    current_checkpoint['checkpoint_b']['start_sample'] = int(match.group(1))
                match = re.search(r'end_sample=(\d+)', line)
                if match:
                    current_checkpoint['checkpoint_b']['end_sample'] = int(match.group(1))

        i += 1

    if current_checkpoint:
        checkpoints.append(current_checkpoint)

    return checkpoints


def compare_checkpoints(python_checkpoints, teensy_checkpoints):
    """Compare Python and Teensy checkpoint values"""

    print_section("Checkpoint Comparison")

    for py_cp in python_checkpoints:
        epoch = py_cp['epoch']

        # Find matching Teensy checkpoint
        teensy_cp = None
        for t in teensy_checkpoints:
            if t['epoch'] == epoch:
                teensy_cp = t
                break

        if not teensy_cp:
            print(f"\nEpoch {epoch}: No Teensy data found")
            continue

        print(f"\n--- Epoch {epoch} ---")

        # Compare Checkpoint A (100Hz preprocessed)
        if 'checkpoint_a' in py_cp and 'checkpoint_a' in teensy_cp:
            py_a = py_cp['checkpoint_a']
            t_a = teensy_cp['checkpoint_a']

            print("Checkpoint A (100Hz preprocessed):")
            if 'mean' in t_a:
                diff = abs(py_a['mean'] - t_a['mean'])
                status = "OK" if diff < 0.1 else "DIFF"
                print(f"  mean: Python={py_a['mean']:.4f}, Teensy={t_a['mean']:.4f}, diff={diff:.4f} [{status}]")
            if 'std' in t_a:
                diff = abs(py_a['std'] - t_a['std'])
                status = "OK" if diff < 0.1 else "DIFF"
                print(f"  std:  Python={py_a['std']:.4f}, Teensy={t_a['std']:.4f}, diff={diff:.4f} [{status}]")
            if 'first_10' in t_a and 'first_10' in py_a:
                py_first = py_a['first_10'][:5]
                t_first = t_a['first_10'][:5]
                print(f"  first_5: Python={[f'{v:.2f}' for v in py_first]}")
                print(f"           Teensy={[f'{v:.2f}' for v in t_first]}")

        # Compare Checkpoint C (Normalization)
        if 'checkpoint_c' in py_cp and 'checkpoint_c' in teensy_cp:
            py_c = py_cp['checkpoint_c']
            t_c = teensy_cp['checkpoint_c']

            print("Checkpoint C (Normalization):")
            if 'mean' in t_c:
                diff = abs(py_c['mean'] - t_c['mean'])
                status = "OK" if diff < 0.001 else "DIFF"
                print(f"  mean: Python={py_c['mean']:.6f}, Teensy={t_c['mean']:.6f}, diff={diff:.6f} [{status}]")
            if 'std' in t_c:
                diff = abs(py_c['std'] - t_c['std'])
                status = "OK" if diff < 0.001 else "DIFF"
                print(f"  std:  Python={py_c['std']:.6f}, Teensy={t_c['std']:.6f}, diff={diff:.6f} [{status}]")
            if 'norm_first_10' in t_c and 'norm_first_10' in py_c:
                py_norm = py_c['norm_first_10'][:5]
                t_norm = t_c['norm_first_10'][:5]
                print(f"  norm_first_5: Python={[f'{v:.4f}' for v in py_norm]}")
                print(f"                Teensy={[f'{v:.4f}' for v in t_norm]}")

        # Compare Checkpoint D (Model Input - FLOAT32)
        if 'checkpoint_d' in py_cp and 'checkpoint_d' in teensy_cp:
            py_d = py_cp['checkpoint_d']
            t_d = teensy_cp['checkpoint_d']

            print("Checkpoint D (Model Input - FLOAT32):")
            if 'input_type' in t_d:
                print(f"  input_type: Teensy={t_d.get('input_type', 'N/A')}")
            if 'float_first_10' in t_d and 'float_first_10' in py_d:
                py_float = py_d['float_first_10'][:5]
                t_float = t_d['float_first_10'][:5]
                max_diff = max(abs(p - t) for p, t in zip(py_float, t_float)) if t_float else 0
                status = "OK" if max_diff < 0.001 else "DIFF"
                print(f"  float_first_5: Python={[f'{v:.4f}' for v in py_float]}")
                print(f"                 Teensy={[f'{v:.4f}' for v in t_float]} [{status}]")
            if 'epoch_scaled' in t_d:
                py_scaled = py_d.get('epoch_scaled', 0)
                t_scaled = t_d.get('epoch_scaled', 0)
                match = abs(py_scaled - t_scaled) < 0.0001
                status = "OK" if match else "MISMATCH"
                print(f"  epoch_scaled: Python={py_scaled:.6f}, Teensy={t_scaled:.6f} [{status}]")


def generate_reference_checkpoints():
    """Generate Python checkpoints using reference EEG data"""

    if not REFERENCE_EEG.exists():
        print("Reference EEG not found, using raw data preprocessing")
        return None

    print_section("Using Reference EEG for Checkpoints")

    ref_eeg = np.load(REFERENCE_EEG)
    print(f"Reference EEG shape: {ref_eeg.shape}")

    checkpoints = generate_python_checkpoints(ref_eeg, num_epochs=10)
    return checkpoints


def main():
    print("#" * 70)
    print("# Phase 2: Teensy vs Python Comparison Tool")
    print("#" * 70)

    # Check command line arguments
    if len(sys.argv) > 1:
        teensy_file = Path(sys.argv[1])
        if not teensy_file.exists():
            print(f"ERROR: Teensy output file not found: {teensy_file}")
            return
    else:
        teensy_file = None
        print("No Teensy output file provided.")
        print("Usage: python tools/compare_teensy_python.py <teensy_debug.txt>")
        print("\nGenerating Python reference checkpoints only...\n")

    # Generate Python checkpoints
    # Option 1: Use reference preprocessed EEG
    ref_checkpoints = generate_reference_checkpoints()

    # Option 2: Preprocess raw data
    if RAW_DATA_FILE.exists():
        data_100hz = preprocess_raw_data()
        raw_checkpoints = generate_python_checkpoints(data_100hz, num_epochs=10)
    else:
        raw_checkpoints = None

    # Print Python checkpoints for reference
    print_section("Python Reference Checkpoints (from Reference EEG)")
    if ref_checkpoints:
        for cp in ref_checkpoints[:5]:
            print(f"\nEpoch {cp['epoch']}:")
            if 'checkpoint_a' in cp:
                a = cp['checkpoint_a']
                print(f"  [A] mean={a['mean']:.4f}, std={a['std']:.4f}")
                print(f"      first_5: {[f'{v:.2f}' for v in a['first_10'][:5]]}")
            if 'checkpoint_c' in cp:
                c = cp['checkpoint_c']
                print(f"  [C] norm_mean={c['mean']:.6f}, norm_std={c['std']:.6f}")
                print(f"      norm_first_5: {[f'{v:.4f}' for v in c['norm_first_10'][:5]]}")
            if 'checkpoint_d' in cp:
                d = cp['checkpoint_d']
                print(f"  [D] input_type={d.get('input_type', 'FLOAT32')}")
                print(f"      float_first_5: {[f'{v:.4f}' for v in d['float_first_10'][:5]]}")
                print(f"      epoch_scaled={d['epoch_scaled']:.6f}")

    # Parse and compare Teensy output if provided
    if teensy_file:
        print_section(f"Parsing Teensy Output: {teensy_file}")
        teensy_checkpoints = parse_teensy_output(teensy_file)
        print(f"Found {len(teensy_checkpoints)} Teensy checkpoints")

        if ref_checkpoints:
            compare_checkpoints(ref_checkpoints, teensy_checkpoints)

    # Summary
    print_section("Summary")
    print("This tool helps identify where Teensy and Python diverge.")
    print("\nTo use with Teensy:")
    print("1. Enable debug output in firmware (uncomment checkpoint logging)")
    print("2. Run firmware and capture serial output to file")
    print("3. Run: python tools/compare_teensy_python.py teensy_debug.txt")
    print("\nExpected checkpoint format from Teensy:")
    print("  [CHECKPOINT A] 100Hz preprocessed - Epoch X")
    print("  mean=X.XXXX std=X.XXXX min=X.XXXX max=X.XXXX")
    print("  first_10: X.XX X.XX ...")


if __name__ == '__main__':
    main()
