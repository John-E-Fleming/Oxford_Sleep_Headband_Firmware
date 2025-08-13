#include "MLInference.h"

MLInference::MLInference() 
  : interpreter_(nullptr), resolver_(nullptr), tensor_arena_(nullptr), initialized_(false) {
}

MLInference::~MLInference() {
  // Simplified destructor for testing
}

bool MLInference::begin() {
  if (initialized_) {
    return true;
  }
  
  Serial.println("ML Inference: TensorFlow Lite temporarily disabled for testing");
  Serial.println("ML Inference: This will return dummy predictions");
  
  initialized_ = true;
  return true;
}

bool MLInference::predict(float* input_data, float* output_data) {
  if (!initialized_) {
    Serial.println("Model not initialized");
    return false;
  }
  
  // Return dummy prediction for testing
  // This will be replaced with actual TensorFlow Lite implementation later
  output_data[0] = 0.7f;  // WAKE
  output_data[1] = 0.1f;  // LIGHT_SLEEP
  output_data[2] = 0.1f;  // DEEP_SLEEP
  output_data[3] = 0.1f;  // REM_SLEEP
  
  return true;
}

SleepStage MLInference::getPredictedStage(float* output_data) {
  int max_index = 0;
  float max_value = output_data[0];
  
  for (int i = 1; i < MODEL_OUTPUT_SIZE; i++) {
    if (output_data[i] > max_value) {
      max_value = output_data[i];
      max_index = i;
    }
  }
  
  return static_cast<SleepStage>(max_index);
}