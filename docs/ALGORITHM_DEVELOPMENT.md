# Algorithm Development Guide

This document describes the fine-tuning pipeline for future algorithm development, including how to train with new data and integrate accelerometer as an additional input modality.

---

## Table of Contents

1. [Current ML Infrastructure](#current-ml-infrastructure)
2. [Fine-Tuning Workflow](#fine-tuning-workflow)
3. [Gap Analysis](#gap-analysis)
4. [Obtaining Ground Truth Labels](#obtaining-ground-truth-labels)
5. [Adding Accelerometer Input](#adding-accelerometer-input)
6. [Implementation Roadmap](#implementation-roadmap)
7. [Verification Plan](#verification-plan)

---

## Current ML Infrastructure

### Overview

The sleep headband uses a TensorFlow Lite model running on Teensy 4.1 for real-time sleep stage classification. The infrastructure supports the complete ML lifecycle from data collection to deployment.

### Component Status

| Component | Status | Details |
|-----------|--------|---------|
| **TFLite Model** | Working | `8_tflite_quantized_model.tflite` (55 KB, FLOAT32) |
| **Keras Model** | Available | `7_keras_model.keras` (462 KB) - for fine-tuning |
| **Model Architecture** | Documented | Separable ResNet CNN (~76K params) |
| **Inference Pipeline** | Working | 81-89% agreement with reference |
| **Preprocessing** | Validated | 4kHz -> 100Hz, bandpass 0.5-30Hz, Z-score norm |
| **Data Collection** | Working | Binary EEG files on SD card |
| **Prediction Logging** | Working | CSV files with probabilities |
| **Python Training Scripts** | Available | `trainer.py`, `5_training_script.py` |

### Model Specifications

```
Input 0 (EEG):     Shape (1, 1, 3000, 1), FLOAT32 - 30s at 100Hz
Input 1 (Epoch):   Shape (1, 1), FLOAT32 - epoch_index / 1000
Output:            Shape (1, 5), FLOAT32 - [Wake, N1, N2, N3, REM] probabilities
```

### Training Configuration

The model was trained with these parameters (from existing scripts):

```python
Architecture: Separable ResNet CNN
Optimizer: Adam (lr=1e-3)
Loss: Categorical cross-entropy
Batch size: 32
Epochs: 5-6
Class weights: [Wake=2.0, N1=1.0, N2=1.0, N3=1.0, REM=1.0]
```

### Key Files

| Purpose | Location |
|---------|----------|
| Original training script | `example_code/.../trainer.py` |
| Model architecture | `example_code/.../nn_model.py` |
| Preprocessing (Python) | `tools/run_inference.py` |
| Keras model (for fine-tuning) | `data/example_datasets/debug/7_keras_model.keras` |
| TFLite model (deployed) | `data/example_datasets/debug/8_tflite_quantized_model.tflite` |
| Firmware inference | `src/MLInference.cpp` |

---

## Fine-Tuning Workflow

### Complete Pipeline Overview

```
+---------------------------------------------------------------------+
|                    FINE-TUNING WORKFLOW                             |
+---------------------------------------------------------------------+
|                                                                     |
|  1. DATA COLLECTION (Device)                                        |
|     +-------------+     +-------------+     +-------------+         |
|     | Raw EEG     | --> | SD Card     | --> | Transfer    |         |
|     | Recording   |     | Storage     |     | to PC       |         |
|     +-------------+     +-------------+     +-------------+         |
|           [OK]                [OK]                [OK]              |
|                                                                     |
|  2. LABELING (Manual or External System)                            |
|     +-------------+     +-------------+     +-------------+         |
|     | PSG/Expert  | --> | Epoch-level | --> | Aligned     |         |
|     | Scoring     |     | Labels      |     | Label File  |         |
|     +-------------+     +-------------+     +-------------+         |
|         [NEEDED]           [NEEDED]           [NEEDED]              |
|                                                                     |
|  3. PREPROCESSING (Python)                                          |
|     +-------------+     +-------------+     +-------------+         |
|     | Load Raw    | --> | Apply       | --> | Create      |         |
|     | Binary      |     | Pipeline    |     | Epochs      |         |
|     +-------------+     +-------------+     +-------------+         |
|           [OK]                [OK]                [OK]              |
|                                                                     |
|  4. TRAINING (Python + TensorFlow)                                  |
|     +-------------+     +-------------+     +-------------+         |
|     | Load Base   | --> | Fine-tune   | --> | Export      |         |
|     | Model       |     | on New Data |     | TFLite      |         |
|     +-------------+     +-------------+     +-------------+         |
|         [PARTIAL]          [PARTIAL]            [OK]                |
|                                                                     |
|  5. DEPLOYMENT (Device)                                             |
|     +-------------+     +-------------+     +-------------+         |
|     | Convert to  | --> | Embed in    | --> | Validate    |         |
|     | model.h     |     | Firmware    |     | Performance |         |
|     +-------------+     +-------------+     +-------------+         |
|           [OK]                [OK]                [OK]              |
|                                                                     |
+---------------------------------------------------------------------+

Legend: [OK] = Available   [PARTIAL] = Needs adaptation   [NEEDED] = Missing
```

### What Works Now

1. **Data Collection**: Device records EEG sessions to SD card in binary format
2. **Preprocessing**: `tools/run_inference.py` applies the validated pipeline
3. **Model Export**: TFLite conversion and `model.h` generation work
4. **Deployment**: Firmware loads and runs TFLite models

### What Needs Implementation

1. **Ground truth labels**: PSG alignment or manual scoring
2. **Training data loader**: Adapt scripts for your binary format + labels
3. **Fine-tuning script**: Modify `trainer.py` for transfer learning

---

## Gap Analysis

### Critical Missing Components

| Gap | Impact | Effort |
|-----|--------|--------|
| **Ground Truth Labels** | Cannot train without labeled data | HIGH - requires sleep expert scoring or PSG integration |
| **Label File Format** | No standard format for epoch labels | LOW - define CSV format |
| **Training Data Loader** | Scripts expect DREEM dataset format | MEDIUM - adapt for new data |
| **Fine-tuning Script** | No script to load pretrained weights | MEDIUM - modify trainer.py |

### Components Needing Adaptation

| Component | Current State | Required Change |
|-----------|---------------|-----------------|
| `trainer.py` | Trains from scratch on DREEM data | Add fine-tuning mode, custom data loader |
| `run_inference.py` | Processes new data format | Good as-is for preprocessing |
| Model export | Works for fresh models | Verify works for fine-tuned models |

---

## Obtaining Ground Truth Labels

### Option A: PSG Integration (Recommended)

If you have access to polysomnography (PSG) equipment:

1. Record with your device simultaneously with PSG
2. Use PSG sleep staging as ground truth
3. Align timestamps between systems
4. Create labeled dataset

**Required tool**: `tools/align_psg_labels.py` (to be implemented)

```python
# tools/align_psg_labels.py
# Input: PSG export (hypnogram), your EEG file
# Output: Aligned epoch labels

Supported formats (to implement):
- EDF+ annotations (common PSG format)
- CSV with timestamp + stage columns
- NSRR XML annotations
```

### Option B: Manual Labeling Tool

Create a Python tool to manually label epochs:

```python
# tools/label_epochs.py
# Displays EEG epochs one at a time
# User selects: Wake, N1, N2, N3, REM
# Saves to labels.csv

Required features:
- Load preprocessed EEG from run_inference.py output
- Display epoch spectrogram + waveform
- Show model prediction as suggestion
- Allow user to confirm/override
- Save epoch labels to CSV
```

**Effort**: ~1-2 days development
**Limitation**: Labor-intensive (1000+ epochs per night)

### Option C: Semi-Supervised Learning (Advanced)

Use model predictions + confidence thresholding:

```python
# Only use epochs where model confidence > 90%
# Treat these as pseudo-labels
# Combine with small amount of manually-labeled data
```

**Risk**: May reinforce model biases. NOT recommended as primary approach.

### Label File Format

Standard format for epoch labels:

```csv
epoch,timestamp_s,label,source
0,30.0,N2,manual
1,60.0,N2,manual
2,90.0,N1,psg
3,120.0,N1,psg
...
```

| Field | Description |
|-------|-------------|
| `epoch` | 0-indexed epoch number |
| `timestamp_s` | Epoch end time in seconds from recording start |
| `label` | Sleep stage: Wake, N1, N2, N3, REM |
| `source` | Label source: manual, psg, model (for pseudo-labels) |

---

## Adding Accelerometer Input

### The Challenge

The current model architecture has fixed input shapes:

```
Input 1: EEG data        -> Shape (1, 3000, 1)
Input 2: Epoch index     -> Shape (1,)
         |
   Separable ResNet CNN
         |
Output: 5-class probabilities
```

**Key constraint**: The original DREEM training data does not include accelerometer recordings. Options requiring full retraining are not feasible.

### Integration Options

| Option | Complexity | Uses Pretrained Weights | Feasibility |
|--------|------------|-------------------------|-------------|
| **A: Late Fusion** | Low | Yes (unchanged) | **Recommended first step** |
| **B: Feature Fusion** | Medium | Partial (frozen EEG) | Viable if A shows value |
| **C: Full Retrain** | High | No | **Not feasible** - original data lacks accelerometer |

### Option A: Late Fusion (Recommended First)

Keep EEG model unchanged, add separate accelerometer classifier:

```
+-------------------------------------------------------------+
|                                                             |
|  EEG Input (3000 samples)                                   |
|       |                                                     |
|  [Existing Model] --------------------------> EEG Probs     |
|                                                   |         |
|  Accel Input (3 channels x 30s)                 Fusion      |
|       |                                           |         |
|  [New Small Model] --------------------------> Final Pred   |
|                                                             |
+-------------------------------------------------------------+
```

**Implementation**:

```python
# Train separate accelerometer classifier
accel_model = Sequential([
    Conv1D(16, 50, activation='relu'),
    MaxPooling1D(10),
    Conv1D(32, 10, activation='relu'),
    GlobalAveragePooling1D(),
    Dense(5, activation='softmax')
])

# At inference time:
eeg_probs = eeg_model.predict(eeg_input)
accel_probs = accel_model.predict(accel_input)
final_probs = 0.7 * eeg_probs + 0.3 * accel_probs  # Weighted average
```

**Pros**:
- Uses existing model unchanged
- Easy to test accelerometer value
- Simple to deploy (run both models)

**Cons**:
- Models don't share features
- May not capture cross-modal patterns

### Option B: Feature Fusion (Transfer Learning)

Freeze EEG encoder, add accelerometer branch, train fusion layers:

```
+-------------------------------------------------------------+
|  EEG Input                  Accel Input                     |
|      |                           |                          |
|  [Frozen CNN]               [New CNN]                       |
|      |                           |                          |
|  EEG Features (64-dim)     Accel Features (32-dim)          |
|      +------------+--------------+                          |
|              Concatenate                                    |
|                   |                                         |
|            [New Dense Layers] <- Trainable                  |
|                   |                                         |
|             5-class output                                  |
+-------------------------------------------------------------+
```

**Implementation**:

```python
# Load pretrained model
base_model = tf.keras.models.load_model('7_keras_model.keras')

# Freeze EEG layers
eeg_encoder = Model(inputs=base_model.input,
                    outputs=base_model.get_layer('global_avg_pool').output)
eeg_encoder.trainable = False

# New accelerometer branch
accel_input = Input(shape=(3000, 3))  # 30s x 3 axes at 100Hz
accel_features = Conv1D(32, 50, activation='relu')(accel_input)
accel_features = GlobalAveragePooling1D()(accel_features)

# Fusion
eeg_features = eeg_encoder([eeg_input, epoch_input])
combined = Concatenate()([eeg_features, accel_features])
output = Dense(32, activation='relu')(combined)
output = Dense(5, activation='softmax')(output)

# Train only new layers
new_model = Model(inputs=[eeg_input, epoch_input, accel_input], outputs=output)
```

**Pros**:
- Preserves EEG knowledge
- Allows cross-modal learning
- Moderate data requirements

**Cons**:
- More complex architecture
- Need to modify firmware for 3-input model

### When Accelerometer Helps Most

Accelerometer is most valuable for:
- **REM detection**: Eye movements during REM
- **Wake detection**: Body movements indicating wakefulness
- **Position changes**: Sleep position affecting EEG quality

### Data Requirements

| Option | Labeled Epochs Needed |
|--------|----------------------|
| Option A (Late Fusion) | ~100-200 epochs to test value |
| Option B (Feature Fusion) | ~500-1000 epochs for good generalization |
| Option C (Full Retrain) | NOT FEASIBLE - original data lacks accelerometer |

---

## Implementation Roadmap

### Phase 1: Data Infrastructure (Can Start Now)

| Task | Effort | Description |
|------|--------|-------------|
| Define label file format | 1 day | CSV schema as documented above |
| Create data loader | 1-2 days | Load binary format + labels for training |
| Test preprocessing | 1 day | Verify epoch shapes match model input |

### Phase 2: Labeling Solution

Choose based on your resources:

| If You Have... | Implement... | Effort |
|----------------|--------------|--------|
| PSG access | PSG alignment tool | 2-3 days |
| Manual only | Labeling GUI | 1-2 days |
| Neither | Semi-supervised approach | 3-5 days |

### Phase 3: Training Pipeline

| Task | Effort | Description |
|------|--------|-------------|
| Modify `trainer.py` | 2-3 days | Accept custom data format |
| Add fine-tuning mode | 1 day | Load pretrained weights |
| Implement validation split | 0.5 day | Hold out test data |
| Test full pipeline | 1 day | Train on small dataset |

### Phase 4: Deployment

| Task | Effort | Description |
|------|--------|-------------|
| Export fine-tuned model | 0.5 day | TFLite conversion |
| Convert to `model.h` | 0.5 day | C array for firmware |
| Test on Teensy | 1 day | Validation playback |
| Compare vs original | 0.5 day | Performance metrics |

### Phase 5: Accelerometer Integration (Optional)

| Task | Effort | Description |
|------|--------|-------------|
| Train accel-only classifier | 1-2 days | Test value of accelerometer |
| Implement late fusion | 1 day | Weighted probability combination |
| Evaluate improvement | 1 day | Compare against EEG-only |
| (If valuable) Feature fusion | 3-5 days | Deep integration |

### Files to Create

```
tools/
├── align_psg_labels.py      # Parse PSG export, align timestamps
├── data_loader.py           # Load binary EEG + labels for training
├── fine_tune.py             # Transfer learning script
├── label_epochs.py          # Manual labeling/correction GUI
└── validate_finetuned.py    # Compare models
```

---

## Verification Plan

### 1. Data Loading Test

Verify that labeled data can be loaded correctly:

```python
X, y = load_training_data('recording.bin', 'labels.csv')
assert X.shape == (n_epochs, 1, 3000, 1)  # Match model input
assert y.shape == (n_epochs, 5)            # One-hot encoded
```

### 2. Fine-tuning Test

Verify training works on small dataset:

```python
model = load_model('7_keras_model.keras')
history = model.fit(X_train, y_train, epochs=1)
assert history.history['loss'][0] > history.history['loss'][-1]  # Loss decreases
```

### 3. Export Test

Verify TFLite conversion succeeds:

```python
converter = tf.lite.TFLiteConverter.from_keras_model(fine_tuned_model)
tflite_model = converter.convert()
assert len(tflite_model) > 0
```

### 4. Deployment Test

Run on Teensy with validation playback:

```
1. Load fine-tuned model in firmware
2. Run playback mode with reference predictions
3. Calculate agreement percentage
```

### 5. Performance Test

Fine-tuned model should match or exceed original:

| Metric | Requirement |
|--------|-------------|
| Overall agreement | >= 85% |
| Per-class accuracy | No class < 70% |
| Inference time | <= 200ms per epoch |

### 6. Accelerometer Test (If Pursued)

Compare REM/Wake detection with and without accelerometer:

| Configuration | REM F1 | Wake F1 |
|---------------|--------|---------|
| EEG only | baseline | baseline |
| EEG + Accel | >= baseline | >= baseline |

---

## Summary

### What You Can Do Now

1. **Collect unlabeled data**: Device already records EEG sessions
2. **Prepare infrastructure**: Define label format, create data loader
3. **Plan labeling strategy**: PSG integration or manual scoring

### What You Need to Implement

| Component | Status | Priority |
|-----------|--------|----------|
| Ground truth labels | Missing | HIGH |
| PSG alignment tool | To build | HIGH (if PSG available) |
| Training data loader | To build | HIGH |
| Fine-tuning script | To build | HIGH |
| Accelerometer model | To build | MEDIUM (after EEG fine-tuning works) |

### Bottom Line

The technical infrastructure for fine-tuning mostly exists:

- Keras model available for transfer learning
- TFLite export pipeline works
- Validation framework in place

The main gap is **labeled training data**. With PSG access and expert scoring, this gap is solvable. Start with the data infrastructure (Phase 1) while arranging for labels.

For accelerometer integration, use **late fusion (Option A)** first to validate that accelerometer adds value before investing in deeper integration.
