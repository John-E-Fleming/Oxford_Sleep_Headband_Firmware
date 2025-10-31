#include "DebugLogger.h"
#include <stdio.h>

// File paths
const char* DebugLogger::FILE_PREPROCESSED = "debug_preprocessed_100hz.csv";
const char* DebugLogger::FILE_NORMALIZED = "debug_normalized.csv";
const char* DebugLogger::FILE_QUANTIZED = "debug_quantized.csv";
const char* DebugLogger::FILE_MODEL_OUTPUT = "debug_model_output.csv";

DebugLogger::DebugLogger()
  : sd_(nullptr), initialized_(false), enabled_(false), row_buffer_(nullptr) {
}

bool DebugLogger::begin(SdFat* sd) {
  if (!sd) {
    Serial.println("DebugLogger: NULL SD pointer");
    return false;
  }

  sd_ = sd;

  // Allocate buffer for building CSV rows
  row_buffer_ = new char[BUFFER_SIZE];
  if (!row_buffer_) {
    Serial.println("DebugLogger: Failed to allocate row buffer");
    return false;
  }

  initialized_ = true;
  Serial.println("DebugLogger: Initialized");
  return true;
}

void DebugLogger::setEnabled(bool enabled) {
  enabled_ = enabled;
  Serial.print("Debug logging: ");
  Serial.println(enabled_ ? "ENABLED" : "DISABLED");
}

bool DebugLogger::fileExists(const char* filename) {
  return sd_->exists(filename);
}

bool DebugLogger::writeHeader(const char* filename, int num_samples) {
  if (!initialized_) return false;

  SdFile file;
  if (!file.open(filename, O_WRONLY | O_CREAT | O_TRUNC)) {
    Serial.print("DebugLogger: Failed to create ");
    Serial.println(filename);
    return false;
  }

  // Write header: Epoch,Sample_0,Sample_1,...,Sample_N
  file.print("Epoch");
  for (int i = 0; i < num_samples; i++) {
    file.print(",Sample_");
    file.print(i);
  }
  file.println();

  file.close();
  return true;
}

bool DebugLogger::appendRow(const char* filename, const char* row) {
  if (!initialized_) return false;

  SdFile file;
  if (!file.open(filename, O_WRONLY | O_CREAT | O_APPEND)) {
    Serial.print("DebugLogger: Failed to open ");
    Serial.println(filename);
    return false;
  }

  file.print(row);
  file.close();
  return true;
}

bool DebugLogger::logPreprocessed100Hz(float* data, int length, int epoch_index) {
  if (!enabled_ || !initialized_) return false;

  // Create header if file doesn't exist
  if (!fileExists(FILE_PREPROCESSED)) {
    writeHeader(FILE_PREPROCESSED, length);
  }

  // Build row: Epoch,val0,val1,...,valN
  int offset = 0;
  offset += snprintf(row_buffer_ + offset, BUFFER_SIZE - offset, "%d", epoch_index);

  for (int i = 0; i < length; i++) {
    if (offset >= BUFFER_SIZE - 20) {
      Serial.println("DebugLogger: Buffer overflow in preprocessed");
      return false;
    }
    offset += snprintf(row_buffer_ + offset, BUFFER_SIZE - offset, ",%.2f", data[i]);
  }
  offset += snprintf(row_buffer_ + offset, BUFFER_SIZE - offset, "\n");

  return appendRow(FILE_PREPROCESSED, row_buffer_);
}

bool DebugLogger::logNormalizedWindow(float* data, int length, int epoch_index) {
  if (!enabled_ || !initialized_) return false;

  // Create header if file doesn't exist
  if (!fileExists(FILE_NORMALIZED)) {
    writeHeader(FILE_NORMALIZED, length);
  }

  // Build row: Epoch,val0,val1,...,valN
  int offset = 0;
  offset += snprintf(row_buffer_ + offset, BUFFER_SIZE - offset, "%d", epoch_index);

  for (int i = 0; i < length; i++) {
    if (offset >= BUFFER_SIZE - 20) {
      Serial.println("DebugLogger: Buffer overflow in normalized");
      return false;
    }
    offset += snprintf(row_buffer_ + offset, BUFFER_SIZE - offset, ",%.6f", data[i]);
  }
  offset += snprintf(row_buffer_ + offset, BUFFER_SIZE - offset, "\n");

  return appendRow(FILE_NORMALIZED, row_buffer_);
}

bool DebugLogger::logQuantizedInput(int8_t* data, int length, int epoch_index) {
  if (!enabled_ || !initialized_) return false;

  // Create header if file doesn't exist
  if (!fileExists(FILE_QUANTIZED)) {
    writeHeader(FILE_QUANTIZED, length);
  }

  // Build row: Epoch,val0,val1,...,valN
  int offset = 0;
  offset += snprintf(row_buffer_ + offset, BUFFER_SIZE - offset, "%d", epoch_index);

  for (int i = 0; i < length; i++) {
    if (offset >= BUFFER_SIZE - 20) {
      Serial.println("DebugLogger: Buffer overflow in quantized");
      return false;
    }
    offset += snprintf(row_buffer_ + offset, BUFFER_SIZE - offset, ",%d", (int)data[i]);
  }
  offset += snprintf(row_buffer_ + offset, BUFFER_SIZE - offset, "\n");

  return appendRow(FILE_QUANTIZED, row_buffer_);
}

bool DebugLogger::logModelOutput(float* output, int output_size, int epoch_index, float time_seconds, const char* predicted_stage) {
  if (!enabled_ || !initialized_) return false;

  // Create header if file doesn't exist
  if (!fileExists(FILE_MODEL_OUTPUT)) {
    SdFile file;
    if (!file.open(FILE_MODEL_OUTPUT, O_WRONLY | O_CREAT | O_TRUNC)) {
      Serial.println("DebugLogger: Failed to create model output file");
      return false;
    }
    file.println("Epoch,Time_s,N3,N2,N1,REM,WAKE,Predicted_Stage");
    file.close();
  }

  // Build row: Epoch,Time,val0,val1,...,valN,Stage
  int offset = 0;
  offset += snprintf(row_buffer_ + offset, BUFFER_SIZE - offset,
                     "%d,%.1f", epoch_index, time_seconds);

  for (int i = 0; i < output_size; i++) {
    offset += snprintf(row_buffer_ + offset, BUFFER_SIZE - offset, ",%.6f", output[i]);
  }
  offset += snprintf(row_buffer_ + offset, BUFFER_SIZE - offset, ",%s\n", predicted_stage);

  return appendRow(FILE_MODEL_OUTPUT, row_buffer_);
}
