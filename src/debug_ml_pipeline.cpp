/**
 * ML Pipeline Debugging Tool for Teensy 4.1
 * ==========================================
 *
 * This test loads reference data from SD card and validates the ML pipeline
 * step-by-step against known-good outputs from a working implementation.
 *
 * Reference files (from data/example_datasets/debug/):
 * - 1_bandpassed_eeg_single_channel.npy (2,880,000 samples, 8 hours @ 100Hz)
 * - 2_standardized_epochs.npy (960 epochs x 3000 samples)
 * - 3_quantized_model_predictions.npy (960 predictions)
 * - 4_quantized_model_probabilities.npy (960 x 5 probabilities)
 * - 8_tflite_quantized_model.tflite (the model)
 *
 * Usage:
 * - Place reference files in SD card root: /debug/
 * - Upload this sketch to Teensy 4.1
 * - Open Serial Monitor
 * - Send commands:
 *   't' - Run full pipeline test
 *   'n' - Test normalization only
 *   'q' - Test quantization only
 *   'i' - Test inference only
 *   's' - Show statistics
 */

#include <Arduino.h>
#include <SdFat.h>
#include <SPI.h>
#include "EEGProcessor.h"
#include "MLInference.h"
#include "model.h"

// SdFat object for SD card access (matching test_eeg_playback.cpp)
SdFat SD;

// Test configuration
#define TEST_NUM_EPOCHS 960  // Test all 960 epochs (8 hours)
#define EPOCH_SIZE 3000      // 30 seconds at 100Hz

// Global objects
EEGProcessor processor;
MLInference mlInference;

// Statistics tracking
struct TestStats {
  int epochs_tested;
  int predictions_matched;
  int predictions_total;
  float max_prob_difference;
  float mean_prob_difference;
  float normalization_error;
};

TestStats test_stats = {0};

/**
 * Load a single float from .npy file at specific offset
 * .npy format: 128 byte header + data
 */
bool loadNumpyFloat(SdFile& file, size_t index, float* value) {
  const size_t NPY_HEADER_SIZE = 128;  // Standard numpy header size
  size_t offset = NPY_HEADER_SIZE + (index * sizeof(float));

  if (!file.seek(offset)) {
    return false;
  }

  if (file.read((uint8_t*)value, sizeof(float)) != sizeof(float)) {
    return false;
  }

  return true;
}

/**
 * Load an epoch (3000 samples) from the bandpassed EEG file
 */
bool loadEEGEpoch(SdFile& file, int epoch_index, float* buffer) {
  const size_t NPY_HEADER_SIZE = 128;
  size_t start_sample = epoch_index * EPOCH_SIZE;
  size_t offset = NPY_HEADER_SIZE + (start_sample * sizeof(float));

  if (!file.seek(offset)) {
    Serial.print("ERROR: Failed to seek to epoch ");
    Serial.print(epoch_index);
    Serial.print(" at offset ");
    Serial.println(offset);
    return false;
  }

  size_t bytes_to_read = EPOCH_SIZE * sizeof(float);
  int bytes_read = file.read((uint8_t*)buffer, bytes_to_read);

  if (bytes_read != (int)bytes_to_read) {
    Serial.print("ERROR: Failed to read epoch ");
    Serial.print(epoch_index);
    Serial.print(" - expected ");
    Serial.print(bytes_to_read);
    Serial.print(" bytes, got ");
    Serial.println(bytes_read);
    return false;
  }

  // Debug: Print first few raw values
  if (epoch_index == 0) {
    Serial.println("DEBUG: First 5 raw float values:");
    for (int i = 0; i < 5; i++) {
      Serial.print("  [");
      Serial.print(i);
      Serial.print("] = ");
      Serial.print(buffer[i], 6);
      Serial.print(" (0x");
      uint32_t* raw = (uint32_t*)&buffer[i];
      Serial.print(*raw, HEX);
      Serial.println(")");
    }
  }

  return true;
}

/**
 * Load reference normalized epoch from file 2
 */
bool loadReferenceNormalized(SdFile& file, int epoch_index, float* buffer) {
  const size_t NPY_HEADER_SIZE = 128;
  size_t offset = NPY_HEADER_SIZE + (epoch_index * EPOCH_SIZE * sizeof(float));

  if (!file.seek(offset)) {
    return false;
  }

  size_t bytes_to_read = EPOCH_SIZE * sizeof(float);
  if (file.read((uint8_t*)buffer, bytes_to_read) != bytes_to_read) {
    return false;
  }

  return true;
}

/**
 * Load reference prediction for an epoch (single int32)
 */
bool loadReferencePrediction(SdFile& file, int epoch_index, int32_t* prediction) {
  const size_t NPY_HEADER_SIZE = 128;
  size_t offset = NPY_HEADER_SIZE + (epoch_index * sizeof(int32_t));

  if (!file.seek(offset)) {
    return false;
  }

  if (file.read((uint8_t*)prediction, sizeof(int32_t)) != sizeof(int32_t)) {
    return false;
  }

  return true;
}

/**
 * Load reference probabilities for an epoch (5 floats)
 */
bool loadReferenceProbabilities(SdFile& file, int epoch_index, float* probs) {
  const size_t NPY_HEADER_SIZE = 128;
  size_t offset = NPY_HEADER_SIZE + (epoch_index * 5 * sizeof(float));

  if (!file.seek(offset)) {
    return false;
  }

  size_t bytes_to_read = 5 * sizeof(float);
  if (file.read((uint8_t*)probs, bytes_to_read) != bytes_to_read) {
    return false;
  }

  return true;
}

// Global file handles to avoid repeated open/close
SdFile g_eeg_file;
SdFile g_norm_file;
SdFile g_pred_file;
SdFile g_prob_file;
bool g_files_open = false;

bool openAllFiles() {
  if (g_files_open) return true;

  if (!g_eeg_file.open("debug/1_bandpassed_eeg_single_channel.npy", FILE_READ)) {
    Serial.println("ERROR: Cannot open EEG file");
    return false;
  }
  if (!g_norm_file.open("debug/2_standardized_epochs.npy", FILE_READ)) {
    Serial.println("ERROR: Cannot open normalized file");
    g_eeg_file.close();
    return false;
  }
  if (!g_pred_file.open("debug/3_quantized_model_predictions.npy", FILE_READ)) {
    Serial.println("ERROR: Cannot open predictions file");
    g_eeg_file.close();
    g_norm_file.close();
    return false;
  }
  if (!g_prob_file.open("debug/4_quantized_model_probabilities.npy", FILE_READ)) {
    Serial.println("ERROR: Cannot open probabilities file");
    g_eeg_file.close();
    g_norm_file.close();
    g_pred_file.close();
    return false;
  }

  g_files_open = true;
  return true;
}

void closeAllFiles() {
  if (!g_files_open) return;
  g_eeg_file.close();
  g_norm_file.close();
  g_pred_file.close();
  g_prob_file.close();
  g_files_open = false;
}

/**
 * Test normalization against reference
 */
void testNormalization(int epoch_index) {
  Serial.println("\n=== Testing Normalization ===");
  Serial.print("Epoch: ");
  Serial.println(epoch_index);

  // Allocate buffers
  float* eeg_raw = new float[EPOCH_SIZE];
  float* eeg_normalized = new float[EPOCH_SIZE];
  float* ref_normalized = new float[EPOCH_SIZE];

  if (!eeg_raw || !eeg_normalized || !ref_normalized) {
    Serial.println("ERROR: Failed to allocate buffers");
    delete[] eeg_raw;
    delete[] eeg_normalized;
    delete[] ref_normalized;
    return;
  }

  if (!openAllFiles()) {
    Serial.println("ERROR: Failed to open reference files");
    delete[] eeg_raw;
    delete[] eeg_normalized;
    delete[] ref_normalized;
    return;
  }

  // Load raw EEG epoch
  if (!loadEEGEpoch(g_eeg_file, epoch_index, eeg_raw)) {
    Serial.println("ERROR: Failed to load EEG epoch");
    delete[] eeg_raw;
    delete[] eeg_normalized;
    delete[] ref_normalized;
    return;
  }

  // Debug: Check if raw data is valid
  bool has_invalid_data = false;
  for (int i = 0; i < 10; i++) {  // Check first 10 samples
    if (isnan(eeg_raw[i]) || isinf(eeg_raw[i])) {
      has_invalid_data = true;
      Serial.print("WARNING: Invalid data at sample ");
      Serial.print(i);
      Serial.print(": ");
      Serial.println(eeg_raw[i]);
    }
  }
  if (has_invalid_data) {
    Serial.println("ERROR: Raw EEG data contains NaN or Inf values!");
  }

  // Load reference normalized
  if (!loadReferenceNormalized(g_norm_file, epoch_index, ref_normalized)) {
    Serial.println("ERROR: Failed to load reference normalized");
    delete[] eeg_raw;
    delete[] eeg_normalized;
    delete[] ref_normalized;
    return;
  }

  Serial.println("✓ Loaded raw and reference data");

  // Calculate statistics on raw epoch
  float epoch_mean = 0.0f;
  for (int i = 0; i < EPOCH_SIZE; i++) {
    epoch_mean += eeg_raw[i];
  }
  epoch_mean /= EPOCH_SIZE;

  float epoch_std = 0.0f;
  for (int i = 0; i < EPOCH_SIZE; i++) {
    float diff = eeg_raw[i] - epoch_mean;
    epoch_std += diff * diff;
  }
  epoch_std = sqrt(epoch_std / EPOCH_SIZE);

  Serial.print("Raw epoch - Mean: ");
  Serial.print(epoch_mean, 6);
  Serial.print(", Std: ");
  Serial.println(epoch_std, 6);

  // Normalize using per-epoch z-score (this is what reference does)
  for (int i = 0; i < EPOCH_SIZE; i++) {
    eeg_normalized[i] = (eeg_raw[i] - epoch_mean) / epoch_std;
  }

  Serial.println("✓ Normalized epoch using per-epoch z-score");

  // Compare with reference
  float max_diff = 0.0f;
  float sum_diff = 0.0f;
  int max_diff_idx = 0;

  for (int i = 0; i < EPOCH_SIZE; i++) {
    float diff = fabs(eeg_normalized[i] - ref_normalized[i]);
    sum_diff += diff;
    if (diff > max_diff) {
      max_diff = diff;
      max_diff_idx = i;
    }
  }

  float mean_diff = sum_diff / EPOCH_SIZE;

  Serial.println("\n--- Comparison Results ---");
  Serial.print("Max difference: ");
  Serial.print(max_diff, 6);
  Serial.print(" at index ");
  Serial.println(max_diff_idx);
  Serial.print("Mean difference: ");
  Serial.println(mean_diff, 6);

  if (mean_diff < 0.001f) {
    Serial.println("✓ PASS: Normalization matches reference!");
  } else {
    Serial.println("✗ FAIL: Normalization differs from reference");
    Serial.println("\nSample comparison (first 10 samples):");
    for (int i = 0; i < 10; i++) {
      Serial.print("  [");
      Serial.print(i);
      Serial.print("] Ours: ");
      Serial.print(eeg_normalized[i], 6);
      Serial.print(" vs Ref: ");
      Serial.print(ref_normalized[i], 6);
      Serial.print(" (diff: ");
      Serial.print(eeg_normalized[i] - ref_normalized[i], 6);
      Serial.println(")");
    }
  }

  test_stats.normalization_error = mean_diff;

  delete[] eeg_raw;
  delete[] eeg_normalized;
  delete[] ref_normalized;
}

/**
 * Test inference against reference
 */
void testInference(int epoch_index) {
  Serial.println("\n=== Testing Inference ===");
  Serial.print("Epoch: ");
  Serial.println(epoch_index);

  // Allocate buffers
  float* ref_normalized = new float[MODEL_INPUT_SIZE];
  float* output_probs = new float[MODEL_OUTPUT_SIZE];
  float* ref_probs = new float[MODEL_OUTPUT_SIZE];

  if (!ref_normalized || !output_probs || !ref_probs) {
    Serial.println("ERROR: Failed to allocate buffers");
    delete[] ref_normalized;
    delete[] output_probs;
    delete[] ref_probs;
    return;
  }

  if (!openAllFiles()) {
    Serial.println("ERROR: Failed to open reference files");
    delete[] ref_normalized;
    delete[] output_probs;
    delete[] ref_probs;
    return;
  }

  // Load reference normalized epoch
  if (!loadReferenceNormalized(g_norm_file, epoch_index, ref_normalized)) {
    Serial.println("ERROR: Failed to load reference normalized");
    delete[] ref_normalized;
    delete[] output_probs;
    delete[] ref_probs;
    return;
  }

  // Load reference prediction
  int32_t ref_prediction;
  if (!loadReferencePrediction(g_pred_file, epoch_index, &ref_prediction)) {
    Serial.println("ERROR: Failed to load reference prediction");
    delete[] ref_normalized;
    delete[] output_probs;
    delete[] ref_probs;
    return;
  }

  // Load reference probabilities
  if (!loadReferenceProbabilities(g_prob_file, epoch_index, ref_probs)) {
    Serial.println("ERROR: Failed to load reference probabilities");
    delete[] ref_normalized;
    delete[] output_probs;
    delete[] ref_probs;
    return;
  }

  Serial.println("✓ Loaded reference data");

  // Run inference using our implementation
  if (!mlInference.predict(ref_normalized, output_probs, epoch_index)) {
    Serial.println("ERROR: Inference failed");
    delete[] ref_normalized;
    delete[] output_probs;
    delete[] ref_probs;
    return;
  }

  Serial.println("✓ Inference complete");

  // Get our prediction
  SleepStage our_stage = mlInference.getPredictedStage(output_probs);

  const char* stage_names[] = {"N3_Deep", "N2_Light", "N1_VeryLight", "REM", "Wake"};

  Serial.println("\n--- Reference Output ---");
  Serial.print("Predicted class: ");
  Serial.print(ref_prediction);
  Serial.print(" (");
  Serial.print(stage_names[ref_prediction]);
  Serial.println(")");
  Serial.println("Probabilities:");
  for (int i = 0; i < MODEL_OUTPUT_SIZE; i++) {
    Serial.print("  ");
    Serial.print(stage_names[i]);
    Serial.print(": ");
    Serial.println(ref_probs[i], 6);
  }

  Serial.println("\n--- Our Output ---");
  Serial.print("Predicted class: ");
  Serial.print(our_stage);
  Serial.print(" (");
  Serial.print(stage_names[our_stage]);
  Serial.println(")");
  Serial.println("Probabilities:");
  for (int i = 0; i < MODEL_OUTPUT_SIZE; i++) {
    Serial.print("  ");
    Serial.print(stage_names[i]);
    Serial.print(": ");
    Serial.println(output_probs[i], 6);
  }

  // Compare results
  Serial.println("\n--- Comparison ---");

  bool pred_match = (our_stage == ref_prediction);
  Serial.print("Prediction match: ");
  Serial.println(pred_match ? "✓ YES" : "✗ NO");

  float max_prob_diff = 0.0f;
  float sum_prob_diff = 0.0f;
  for (int i = 0; i < MODEL_OUTPUT_SIZE; i++) {
    float diff = fabs(output_probs[i] - ref_probs[i]);
    sum_prob_diff += diff;
    if (diff > max_prob_diff) {
      max_prob_diff = diff;
    }
  }
  float mean_prob_diff = sum_prob_diff / MODEL_OUTPUT_SIZE;

  Serial.print("Max probability difference: ");
  Serial.println(max_prob_diff, 6);
  Serial.print("Mean probability difference: ");
  Serial.println(mean_prob_diff, 6);

  // Update statistics
  test_stats.predictions_total++;
  if (pred_match) {
    test_stats.predictions_matched++;
  }
  if (max_prob_diff > test_stats.max_prob_difference) {
    test_stats.max_prob_difference = max_prob_diff;
  }
  test_stats.mean_prob_difference += mean_prob_diff;

  delete[] ref_normalized;
  delete[] output_probs;
  delete[] ref_probs;
}

/**
 * Helper function to print separator line
 */
void printSeparator(char c, int length) {
  Serial.println();
  for (int i = 0; i < length; i++) {
    Serial.print(c);
  }
  Serial.println();
}

/**
 * Run full pipeline test on multiple epochs
 */
void runQuickTest() {
  printSeparator('=', 60);
  Serial.println("Running Quick Test (First 20 Epochs)");
  printSeparator('=', 60);

  // Reset statistics
  test_stats = {0};

  // Open all files once
  if (!openAllFiles()) {
    Serial.println("ERROR: Failed to open data files");
    return;
  }

  // Test first 20 epochs for quick validation
  for (int epoch = 0; epoch < 20; epoch++) {
    Serial.print("Testing Epoch ");
    Serial.print(epoch);
    Serial.print(" (Time: ");
    Serial.print(epoch * 30);
    Serial.print("s - ");
    Serial.print((epoch + 1) * 30);
    Serial.println("s)");

    // Only test inference (which uses pre-normalized data from file #2)
    testInference(epoch);

    test_stats.epochs_tested++;
  }

  // Close all files
  closeAllFiles();

  // Print summary
  printSeparator('=', 60);
  Serial.println("Quick Test Summary");
  printSeparator('=', 60);
  Serial.print("Epochs tested: ");
  Serial.println(test_stats.epochs_tested);

  if (test_stats.predictions_total > 0) {
    Serial.print("Predictions matched: ");
    Serial.print(test_stats.predictions_matched);
    Serial.print(" / ");
    Serial.print(test_stats.predictions_total);
    Serial.print(" (");
    Serial.print(100.0f * test_stats.predictions_matched / test_stats.predictions_total);
    Serial.println("%)");
    Serial.print("Max probability difference: ");
    Serial.println(test_stats.max_prob_difference, 6);
    Serial.print("Mean probability difference: ");
    Serial.println(test_stats.mean_prob_difference / test_stats.epochs_tested, 6);
  }

  printSeparator('=', 60);
}

void runFullTest() {
  printSeparator('=', 60);
  Serial.println("Running Full Pipeline Test (All 960 Epochs)");
  printSeparator('=', 60);

  // Reset statistics
  test_stats = {0};

  // Open all files once
  if (!openAllFiles()) {
    Serial.println("ERROR: Failed to open data files");
    return;
  }

  // Test first N epochs
  for (int epoch = 0; epoch < TEST_NUM_EPOCHS; epoch++) {
    // Print progress every 50 epochs
    if (epoch % 50 == 0 || epoch < 10) {
      printSeparator('-', 60);
      Serial.print("Testing Epoch ");
      Serial.print(epoch);
      Serial.print(" (Time: ");
      Serial.print(epoch * 30);
      Serial.print("s - ");
      Serial.print((epoch + 1) * 30);
      Serial.println("s)");
      printSeparator('-', 60);
    } else if (epoch % 10 == 0) {
      Serial.print("Epoch ");
      Serial.print(epoch);
      Serial.println("...");
    }

    // Skip normalization test - file #1 has scaling issues
    // testNormalization(epoch);

    // Only test inference (which uses pre-normalized data from file #2)
    testInference(epoch);

    test_stats.epochs_tested++;
  }

  // Close all files
  closeAllFiles();

  // Print summary
  printSeparator('=', 60);
  Serial.println("Test Summary");
  printSeparator('=', 60);
  Serial.print("Epochs tested: ");
  Serial.println(test_stats.epochs_tested);
  Serial.print("Predictions matched: ");
  Serial.print(test_stats.predictions_matched);
  Serial.print(" / ");
  Serial.print(test_stats.predictions_total);
  Serial.print(" (");
  Serial.print(100.0f * test_stats.predictions_matched / test_stats.predictions_total);
  Serial.println("%)");
  Serial.print("Max probability difference: ");
  Serial.println(test_stats.max_prob_difference, 6);
  Serial.print("Mean probability difference: ");
  Serial.println(test_stats.mean_prob_difference / test_stats.epochs_tested, 6);
  Serial.print("Mean normalization error: ");
  Serial.println(test_stats.normalization_error, 6);
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000);  // Wait up to 3 seconds for serial

  Serial.println("\n===========================================");
  Serial.println("ML Pipeline Debugging Tool");
  Serial.println("===========================================");

  // Initialize SD card using SdFat library (matching test_eeg_playback.cpp)
  Serial.print("Initializing SD card...");
  bool sd_ok = false;

  // Method 1: SdioConfig with FIFO (exactly like test_eeg_playback.cpp)
  if (SD.begin(SdioConfig(FIFO_SDIO))) {
    sd_ok = true;
    Serial.println(" OK (FIFO_SDIO)");
  }
  // Method 2: Fallback to DMA SDIO
  else if (SD.begin(SdioConfig(DMA_SDIO))) {
    sd_ok = true;
    Serial.println(" OK (DMA_SDIO)");
  }
  // Method 3: SPI fallback
  else if (SD.begin(SdSpiConfig(10, DEDICATED_SPI, SD_SCK_MHZ(50)))) {
    sd_ok = true;
    Serial.println(" OK (SPI)");
  }

  if (!sd_ok) {
    Serial.println(" FAILED!");
    Serial.println("Please insert SD card with debug files in /debug/ folder");
    Serial.println("Make sure SD card is properly inserted in Teensy 4.1 slot");
    return;
  }

  // Check for debug folder (note: no leading slash for SD library compatibility)
  if (!SD.exists("debug")) {
    Serial.println("ERROR: debug folder not found on SD card");
    Serial.println("Please create debug/ folder and copy reference files");
    return;
  }

  // List files in debug directory
  Serial.println("\nChecking debug folder contents:");
  SdFile debugDir;
  if (debugDir.open("debug", O_RDONLY)) {
    while (true) {
      SdFile entry;
      if (!entry.openNext(&debugDir, O_RDONLY)) break;

      char name[64];
      entry.getName(name, sizeof(name));
      Serial.print("  Found: ");
      Serial.print(name);
      Serial.print(" (");
      Serial.print(entry.size());
      Serial.println(" bytes)");
      entry.close();
    }
    debugDir.close();
  } else {
    Serial.println("ERROR: Could not open /debug folder");
    return;
  }

  // Check for required files (note: no leading slash for SD library compatibility)
  Serial.println("\nChecking for required files:");
  const char* required_files[] = {
    "debug/1_bandpassed_eeg_single_channel.npy",
    "debug/2_standardized_epochs.npy",
    "debug/3_quantized_model_predictions.npy",
    "debug/4_quantized_model_probabilities.npy"
  };

  bool all_files_exist = true;
  for (int i = 0; i < 4; i++) {
    if (SD.exists(required_files[i])) {
      SdFile f;
      if (f.open(required_files[i], FILE_READ)) {
        Serial.print("  ✓ ");
        Serial.print(required_files[i]);
        Serial.print(" (");
        Serial.print(f.size());
        Serial.println(" bytes)");
        f.close();
      } else {
        Serial.print("  ✗ Cannot open: ");
        Serial.println(required_files[i]);
        all_files_exist = false;
      }
    } else {
      Serial.print("  ✗ MISSING: ");
      Serial.println(required_files[i]);
      all_files_exist = false;
    }
  }

  if (!all_files_exist) {
    Serial.println("\nERROR: Some required files are missing!");
    Serial.println("Please copy all .npy files from:");
    Serial.println("  data/example_datasets/debug/");
    Serial.println("to SD card:");
    Serial.println("  /debug/");
    return;
  }

  Serial.println("✓ All required files found on SD card");

  // Initialize EEG processor
  Serial.print("Initializing EEG processor...");
  if (!processor.begin()) {
    Serial.println(" FAILED!");
    return;
  }
  Serial.println(" OK");

  // Initialize ML inference
  Serial.print("Initializing ML inference...");
  if (!mlInference.begin(false)) {  // Use real model, not dummy
    Serial.println(" FAILED!");
    return;
  }
  Serial.println(" OK");

  Serial.println("\n===========================================");
  Serial.println("Ready! Available commands:");
  Serial.println("  't' - Run full pipeline test (all 960 epochs)");
  Serial.println("  'n' - Test normalization only (epoch 0)");
  Serial.println("  'i' - Test inference only (epoch 0)");
  Serial.println("  's' - Show statistics");
  Serial.println("  'd' - Show SD card diagnostics");
  Serial.println("===========================================\n");
}

void showDiagnostics() {
  Serial.println("\n===========================================");
  Serial.println("SD Card Diagnostics");
  Serial.println("===========================================");

  // Check SD card using same method as setup
  Serial.print("SD card status: ");
  bool sd_ok = false;

  if (SD.begin(SdioConfig(FIFO_SDIO))) {
    sd_ok = true;
    Serial.println("OK (FIFO_SDIO)");
  } else if (SD.begin(SdioConfig(DMA_SDIO))) {
    sd_ok = true;
    Serial.println("OK (DMA_SDIO)");
  } else if (SD.begin(SdSpiConfig(10, DEDICATED_SPI, SD_SCK_MHZ(50)))) {
    sd_ok = true;
    Serial.println("OK (SPI)");
  }

  if (!sd_ok) {
    Serial.println("FAILED - Cannot initialize SD card");
    Serial.println("Check:");
    Serial.println("  - SD card is inserted in Teensy 4.1 built-in slot");
    Serial.println("  - SD card is formatted as FAT32 or exFAT");
    Serial.println("  - SD card is not write-protected");
    return;
  }

  // Check for debug folder
  Serial.print("Debug folder exists: ");
  if (SD.exists("debug")) {
    Serial.println("YES");
  } else {
    Serial.println("NO - Please create 'debug' folder on SD card root");
    return;
  }

  // List all files in debug folder
  Serial.println("\nFiles in debug folder:");
  SdFile debugDir;
  if (debugDir.open("debug", O_RDONLY)) {
    int fileCount = 0;
    while (true) {
      SdFile entry;
      if (!entry.openNext(&debugDir, O_RDONLY)) break;

      char name[64];
      entry.getName(name, sizeof(name));
      Serial.print("  ");
      Serial.print(++fileCount);
      Serial.print(". ");
      Serial.print(name);
      Serial.print(" (");
      Serial.print(entry.size());
      Serial.println(" bytes)");
      entry.close();
    }
    debugDir.close();

    if (fileCount == 0) {
      Serial.println("  (folder is empty)");
    }
  } else {
    Serial.println("  ERROR: Cannot open debug folder");
    return;
  }

  // Check for required files
  Serial.println("\nRequired files check:");
  const char* required_files[] = {
    "debug/1_bandpassed_eeg_single_channel.npy",
    "debug/2_standardized_epochs.npy",
    "debug/3_quantized_model_predictions.npy",
    "debug/4_quantized_model_probabilities.npy"
  };

  for (int i = 0; i < 4; i++) {
    Serial.print("  ");
    if (SD.exists(required_files[i])) {
      SdFile f;
      if (f.open(required_files[i], FILE_READ)) {
        Serial.print("[OK] ");
        Serial.print(required_files[i]);
        Serial.print(" (");
        Serial.print(f.size());
        Serial.println(" bytes)");
        f.close();
      } else {
        Serial.print("[FAIL] Cannot open ");
        Serial.println(required_files[i]);
      }
    } else {
      Serial.print("[MISSING] ");
      Serial.println(required_files[i]);
    }
  }

  Serial.println("\n===========================================");
}

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();

    switch (cmd) {
      case 'q':
      case 'Q':
        runQuickTest();
        break;

      case 't':
      case 'T':
        runFullTest();
        break;

      case 'n':
      case 'N':
        testNormalization(0);
        break;

      case 'i':
      case 'I':
        testInference(0);
        break;

      case 'd':
      case 'D':
        showDiagnostics();
        break;

      case 's':
      case 'S':
        Serial.println("\n=== Current Statistics ===");
        Serial.print("Epochs tested: ");
        Serial.println(test_stats.epochs_tested);
        if (test_stats.predictions_total > 0) {
          Serial.print("Predictions matched: ");
          Serial.print(test_stats.predictions_matched);
          Serial.print(" / ");
          Serial.print(test_stats.predictions_total);
          Serial.print(" (");
          Serial.print(100.0f * test_stats.predictions_matched / test_stats.predictions_total);
          Serial.println("%)");
          Serial.print("Max probability difference: ");
          Serial.println(test_stats.max_prob_difference, 6);
          Serial.print("Mean probability difference: ");
          Serial.println(test_stats.mean_prob_difference / test_stats.epochs_tested, 6);
        }
        break;

      case 'h':
      case 'H':
      case '?':
        Serial.println("\n===========================================");
        Serial.println("Available Commands:");
        Serial.println("===========================================");
        Serial.println("  q - Quick test (first 20 epochs) *** RECOMMENDED ***");
        Serial.println("  t - Full pipeline test (all 960 epochs)");
        Serial.println("  n - Test normalization only (epoch 0)");
        Serial.println("  i - Test inference only (epoch 0)");
        Serial.println("  d - Show SD card diagnostics");
        Serial.println("  s - Show statistics");
        Serial.println("  h - Show this help");
        Serial.println("===========================================");
        break;

      default:
        // Ignore
        break;
    }
  }
}
