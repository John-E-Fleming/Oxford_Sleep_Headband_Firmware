# Python Testing Environment Setup

This guide shows how to set up an isolated Python environment for testing the ML pipeline.

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)

## Setup

### 1. Create Virtual Environment

```bash
# Navigate to project directory
cd sleep_headband_firmware

# Create virtual environment
python -m venv venv
```

### 2. Activate Virtual Environment

**Windows (Command Prompt):**
```bash
venv\Scripts\activate
```

**Windows (PowerShell):**
```bash
venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

You should see `(venv)` prefix in your terminal prompt.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `numpy` - For numerical operations
- `tensorflow` - For TFLite model inference

### 4. Run Tests

```bash
python test_ml_pipeline.py
```

## Deactivating Environment

When you're done testing:

```bash
deactivate
```

## What's Ignored

The `.gitignore` file excludes:
- `venv/` - Virtual environment folder
- `__pycache__/` - Python cache files
- `debug_outputs/` - Test output directory
- `*.npy` - Generated numpy array files (outside data/)
- `*.log` - Log files

This keeps your git repository clean and prevents large binary files from being committed.

## Troubleshooting

### PowerShell Execution Policy Error

If you get an error activating on PowerShell:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### TensorFlow Installation Issues

If TensorFlow installation fails:
- Make sure you have Python 3.9-3.12 (TensorFlow doesn't support 3.13 yet on some platforms)
- Try: `pip install --upgrade pip`
- Try: `pip install tensorflow --no-cache-dir`

### Missing Reference Files

Make sure reference files exist in `data/example_datasets/debug/`:
- `1_bandpassed_eeg_single_channel.npy`
- `2_standardized_epochs.npy`
- `3_quantized_model_predictions.npy`
- `4_quantized_model_probabilities.npy`
- `8_tflite_quantized_model.tflite`

## Quick Reference

```bash
# One-time setup
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Each testing session
venv\Scripts\activate  # Windows
python test_ml_pipeline.py
deactivate
```
