#!/usr/bin/env python3
"""
Pipeline Validation Script

Compares preprocessed EEG data from updated firmware against colleague's reference data
to verify that the preprocessing pipeline produces equivalent outputs.

Usage:
    python validate_pipeline.py --debug-dir /path/to/sd/card --reference-dir example_datasets/debug
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Configuration constants (from config.txt)
VREF = 4.5  # Volts
GAIN = 24
ADC_BITS = 24

def adc_to_uv(adc_value):
    """Convert ADC value to microvolts"""
    return adc_value * (2 * VREF / GAIN) / (2 ** ADC_BITS) * 1e6

def load_debug_data(debug_dir):
    """Load CSV debug files from firmware output"""
    debug_path = Path(debug_dir)

    data = {}

    # Load preprocessed 100Hz data
    preprocessed_file = debug_path / "debug_preprocessed_100hz.csv"
    if preprocessed_file.exists():
        df = pd.read_csv(preprocessed_file)
        epochs = df['Epoch'].values
        data_cols = [col for col in df.columns if col.startswith('Sample_')]
        data['preprocessed'] = df[data_cols].values  # Shape: (n_epochs, 3000)
        data['epochs'] = epochs
        print(f"Loaded preprocessed data: {data['preprocessed'].shape[0]} epochs")
    else:
        print(f"Warning: {preprocessed_file} not found")

    # Load normalized data
    normalized_file = debug_path / "debug_normalized.csv"
    if normalized_file.exists():
        df = pd.read_csv(normalized_file)
        data_cols = [col for col in df.columns if col.startswith('Sample_')]
        data['normalized'] = df[data_cols].values
        print(f"Loaded normalized data: {data['normalized'].shape[0]} epochs")
    else:
        print(f"Warning: {normalized_file} not found")

    # Load quantized data
    quantized_file = debug_path / "debug_quantized.csv"
    if quantized_file.exists():
        df = pd.read_csv(quantized_file)
        data_cols = [col for col in df.columns if col.startswith('Sample_')]
        data['quantized'] = df[data_cols].values
        print(f"Loaded quantized data: {data['quantized'].shape[0]} epochs")
    else:
        print(f"Warning: {quantized_file} not found")

    # Load model output
    model_output_file = debug_path / "debug_model_output.csv"
    if model_output_file.exists():
        data['model_output'] = pd.read_csv(model_output_file)
        print(f"Loaded model output: {len(data['model_output'])} epochs")
    else:
        print(f"Warning: {model_output_file} not found")

    return data

def load_reference_data(reference_dir):
    """Load colleague's reference data"""
    ref_path = Path(reference_dir)

    data = {}

    # TODO: Add loading logic for colleague's reference files
    # This will depend on the format of the files in example_datasets/debug
    print(f"Loading reference data from {ref_path}")

    # Example: Load bandpass filtered data (in ADC steps)
    # filtered_file = ref_path / "filtered_data.csv"  # or .bin or whatever format
    # if filtered_file.exists():
    #     # Load and convert ADC to uV
    #     adc_data = load_file(filtered_file)
    #     data['filtered_uv'] = adc_to_uv(adc_data)

    return data

def calculate_statistics(your_data, reference_data):
    """Calculate comparison statistics between your data and reference"""
    stats = {}

    # Mean Squared Error
    mse = np.mean((your_data - reference_data) ** 2)
    stats['mse'] = mse
    stats['rmse'] = np.sqrt(mse)

    # Correlation
    correlation = np.corrcoef(your_data.flatten(), reference_data.flatten())[0, 1]
    stats['correlation'] = correlation

    # Mean Absolute Error
    mae = np.mean(np.abs(your_data - reference_data))
    stats['mae'] = mae

    # Maximum Absolute Error
    max_error = np.max(np.abs(your_data - reference_data))
    stats['max_error'] = max_error

    # Relative Error
    relative_error = mae / (np.mean(np.abs(reference_data)) + 1e-10)
    stats['relative_error_percent'] = relative_error * 100

    return stats

def plot_comparison(your_data, reference_data, epoch_idx=0, title="Data Comparison", ylabel="Value"):
    """Plot overlay comparison for a specific epoch"""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(your_data[epoch_idx, :], label='Your Implementation', alpha=0.7)
    plt.plot(reference_data[epoch_idx, :], label='Reference (Colleague)', alpha=0.7, linestyle='--')
    plt.xlabel('Sample')
    plt.ylabel(ylabel)
    plt.title(f'{title} - Epoch {epoch_idx}')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    difference = your_data[epoch_idx, :] - reference_data[epoch_idx, :]
    plt.plot(difference, color='red', alpha=0.7)
    plt.xlabel('Sample')
    plt.ylabel('Difference')
    plt.title(f'Difference - Epoch {epoch_idx}')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}_epoch_{epoch_idx}.png')
    print(f"Saved plot: {title.replace(' ', '_')}_epoch_{epoch_idx}.png")
    plt.close()

def generate_report(debug_data, reference_data, stats):
    """Generate validation report"""
    print("\n" + "=" * 60)
    print("PIPELINE VALIDATION REPORT")
    print("=" * 60)

    print(f"\nData loaded:")
    print(f"  Your epochs: {debug_data.get('epochs', [])}")
    print(f"  Preprocessed shape: {debug_data.get('preprocessed', np.array([])).shape}")
    print(f"  Normalized shape: {debug_data.get('normalized', np.array([])).shape}")

    if debug_data.get('model_output') is not None:
        print(f"\nModel Output:")
        print(debug_data['model_output'])

    if stats:
        print(f"\nComparison Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.6f}")

        # Assessment
        print(f"\n" + "-" * 60)
        print("ASSESSMENT:")
        if stats['correlation'] > 0.99:
            print("✅ EXCELLENT: Correlation > 0.99 - pipelines match very well")
        elif stats['correlation'] > 0.95:
            print("✓ GOOD: Correlation > 0.95 - minor differences")
        else:
            print("⚠ WARNING: Correlation < 0.95 - significant differences detected")

        if stats['relative_error_percent'] < 1.0:
            print("✅ EXCELLENT: Relative error < 1%")
        elif stats['relative_error_percent'] < 5.0:
            print("✓ ACCEPTABLE: Relative error < 5%")
        else:
            print("⚠ WARNING: Relative error > 5%")
        print("-" * 60)

    print("\n" + "=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Validate EEG preprocessing pipeline')
    parser.add_argument('--debug-dir', type=str, default='.',
                       help='Directory containing debug CSV files from firmware')
    parser.add_argument('--reference-dir', type=str, default='example_datasets/debug',
                       help='Directory containing reference data from colleague')
    parser.add_argument('--epoch', type=int, default=0,
                       help='Epoch index to plot (default: 0)')
    parser.add_argument('--plot-all', action='store_true',
                       help='Plot all available epochs')

    args = parser.parse_args()

    # Load data
    print("Loading debug data from firmware...")
    debug_data = load_debug_data(args.debug_dir)

    print("\nLoading reference data...")
    reference_data = load_reference_data(args.reference_dir)

    # Calculate statistics if both datasets available
    stats = None
    if 'preprocessed' in debug_data and 'filtered_uv' in reference_data:
        print("\nCalculating comparison statistics...")
        stats = calculate_statistics(debug_data['preprocessed'], reference_data['filtered_uv'])

        # Plot comparison
        if args.plot_all:
            n_epochs = min(debug_data['preprocessed'].shape[0], reference_data['filtered_uv'].shape[0])
            for epoch_idx in range(n_epochs):
                plot_comparison(debug_data['preprocessed'], reference_data['filtered_uv'],
                              epoch_idx, "Preprocessed 100Hz Data", "Amplitude (μV)")
        else:
            plot_comparison(debug_data['preprocessed'], reference_data['filtered_uv'],
                          args.epoch, "Preprocessed 100Hz Data", "Amplitude (μV)")

    # Plot normalized data (self-comparison across epochs)
    if 'normalized' in debug_data:
        plt.figure(figsize=(15, 5))
        for epoch_idx in range(min(3, debug_data['normalized'].shape[0])):
            plt.plot(debug_data['normalized'][epoch_idx, :200], label=f'Epoch {epoch_idx}', alpha=0.7)
        plt.xlabel('Sample')
        plt.ylabel('Normalized Value')
        plt.title('Normalized Data (First 200 samples)')
        plt.legend()
        plt.grid(True)
        plt.savefig('normalized_data_overview.png')
        print("Saved plot: normalized_data_overview.png")
        plt.close()

    # Generate report
    generate_report(debug_data, reference_data, stats)

if __name__ == '__main__':
    main()
