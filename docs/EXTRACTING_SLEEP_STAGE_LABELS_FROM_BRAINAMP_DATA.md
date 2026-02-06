# Extracting Sleep Stage Labels from BrainAmp Data

## Overview
This guide covers how to load BrainAmp data files and extract sleep stage annotations to use as ground truth for algorithm comparison.

## BrainAmp File Structure

BrainAmp systems generate three key files:
- **`.vhdr`** - Header file (text format with metadata)
- **`.eeg`** - Raw EEG data (binary format)
- **`.vmrk`** - Marker/annotation file (contains sleep stage labels)

**Important:** Sleep stage labels are stored in the `.vmrk` file as annotations/markers.

## Required Python Libraries
```bash
pip install mne numpy pandas
```

## Step 1: Load the Data
```python
import mne
import numpy as np
import pandas as pd

# Load the raw BrainAmp data using the .vhdr file
raw = mne.io.read_raw_brainvision('path/to/your_file.vhdr', preload=True)

# Extract annotations (includes sleep stage labels)
annotations = raw.annotations

print(f"Total recording duration: {raw.times[-1]:.2f} seconds")
print(f"Number of annotations: {len(annotations)}")
```

## Step 2: Inspect Annotation Format
```python
# Display first few annotations to understand the format
for i, annot in enumerate(annotations[:10]):
    print(f"Onset: {annot['onset']:.2f}s | Duration: {annot['duration']:.2f}s | Description: {annot['description']}")
```

**Common sleep stage annotation formats:**
- `"Sleep stage W"` / `"Sleep stage N1"` / `"Sleep stage N2"` / `"Sleep stage N3"` / `"Sleep stage R"`
- `"Stage 0"` / `"Stage 1"` / `"Stage 2"` / `"Stage 3"` / `"Stage 5"`
- Check your specific format and adjust parsing accordingly

## Step 3: Parse Sleep Stage Labels
```python
def parse_sleep_stage(description):
    """
    Convert annotation description to numerical sleep stage.
    Adjust mapping based on your annotation format.
    """
    # AASM standard mapping
    stage_map = {
        'W': 0,     # Wake
        'N1': 1,    # NREM Stage 1
        'N2': 2,    # NREM Stage 2
        'N3': 3,    # NREM Stage 3
        'R': 5,     # REM
        'REM': 5,   # REM (alternative)
    }
    
    # Extract stage identifier from description
    for key in stage_map:
        if key in description.upper():
            return stage_map[key]
    
    return -1  # Unknown/unscored

# Test parsing
for annot in annotations[:5]:
    stage = parse_sleep_stage(annot['description'])
    print(f"{annot['description']} -> Stage {stage}")
```

## Step 4: Convert to Epoch-Based Labels

Sleep stages are typically scored in **30-second epochs** (sometimes 20s in older protocols).
```python
# Parameters
EPOCH_DURATION = 30  # seconds (confirm this matches your scoring protocol)

# Calculate total number of epochs
total_duration = raw.times[-1]
n_epochs = int(np.ceil(total_duration / EPOCH_DURATION))

print(f"Total epochs: {n_epochs}")

# Initialize ground truth array
ground_truth = np.full(n_epochs, -1, dtype=int)  # -1 = unscored

# Map annotations to epochs
for annot in annotations:
    description = annot['description']
    
    # Check if this is a sleep stage annotation
    if 'sleep' in description.lower() or 'stage' in description.lower():
        onset_time = annot['onset']
        duration = annot['duration'] if annot['duration'] > 0 else EPOCH_DURATION
        
        # Calculate epoch indices covered by this annotation
        start_epoch = int(onset_time / EPOCH_DURATION)
        end_epoch = int((onset_time + duration) / EPOCH_DURATION)
        
        # Parse the stage
        stage = parse_sleep_stage(description)
        
        # Assign stage to all covered epochs
        for epoch_idx in range(start_epoch, min(end_epoch + 1, n_epochs)):
            ground_truth[epoch_idx] = stage

print(f"Scored epochs: {np.sum(ground_truth >= 0)}")
print(f"Unscored epochs: {np.sum(ground_truth == -1)}")
```

## Step 5: Validate and Export
```python
# Display stage distribution
unique, counts = np.unique(ground_truth[ground_truth >= 0], return_counts=True)
stage_names = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 5: 'REM'}

print("\nSleep stage distribution:")
for stage, count in zip(unique, counts):
    stage_name = stage_names.get(stage, f'Unknown ({stage})')
    percentage = (count / len(ground_truth[ground_truth >= 0])) * 100
    print(f"  {stage_name}: {count} epochs ({percentage:.1f}%)")

# Save ground truth labels
np.save('ground_truth_labels.npy', ground_truth)

# Alternative: save as CSV for easier inspection
df = pd.DataFrame({
    'epoch': np.arange(n_epochs),
    'time_seconds': np.arange(n_epochs) * EPOCH_DURATION,
    'sleep_stage': ground_truth
})
df.to_csv('ground_truth_labels.csv', index=False)

print("\nGround truth labels saved!")
```

## Step 6: Prepare for Algorithm Comparison
```python
# Remove unscored epochs for fair comparison
valid_mask = ground_truth >= 0
ground_truth_clean = ground_truth[valid_mask]

print(f"Clean ground truth shape: {ground_truth_clean.shape}")
print(f"Ready for comparison with your algorithm's predictions")

# Your algorithm predictions should have the same shape
# algorithm_predictions = your_sleep_classifier(data)
# 
# Then compare:
# from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
# 
# kappa = cohen_kappa_score(ground_truth_clean, algorithm_predictions[valid_mask])
# print(f"Cohen's Kappa: {kappa:.3f}")
```

## Key Considerations

### Annotation Format Variations
- **Check your specific format:** Annotation descriptions vary between labs/scorers
- **Modify `parse_sleep_stage()` accordingly**
- Common variations: "Sleep stage W", "Stage 0", "W", "Wake"

### Epoch Duration
- **Standard is 30 seconds** (AASM guidelines)
- **Older studies may use 20 seconds** (Rechtschaffen & Kales)
- **Verify with your data documentation**

### Staging Criteria
- **AASM:** W, N1, N2, N3, R (current standard)
- **R&K:** Stages 0, 1, 2, 3, 4, REM (older standard)
- **Note:** AASM combines R&K stages 3+4 into N3

### Edge Cases
- **Unscored epochs:** Handle gaps in annotations
- **Boundary epochs:** Ensure correct temporal alignment
- **Recording artifacts:** May have "artifact" or "movement" annotations

## Troubleshooting

**Problem:** No sleep stage annotations found
- Check if annotations are in a separate file
- Verify the annotation description format
- Some files may have stages in a separate scoring file

**Problem:** Epoch count mismatch
- Verify EPOCH_DURATION matches scoring protocol
- Check for gaps in recording
- Confirm total duration calculation

**Problem:** Unexpected stage values
- Print all unique annotation descriptions
- Adjust `parse_sleep_stage()` mapping
- Check for non-standard scoring conventions

## Next Steps

Once you have extracted ground truth labels:
1. Ensure your algorithm outputs match the epoch structure
2. Apply the same preprocessing to both datasets
3. Calculate performance metrics (accuracy, kappa, per-class F1)
4. Generate confusion matrix for detailed analysis

## Example Complete Script

See the sections above combined into a single workflow. Adjust file paths and parsing logic based on your specific BrainAmp data format.