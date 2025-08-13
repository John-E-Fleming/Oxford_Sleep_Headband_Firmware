#pragma once

#include <Arduino.h>
#include "model.h"

// Forward declarations - temporarily disabled for testing
// namespace tflite {
//   class MicroInterpreter;
//   template<unsigned int tOpCount>
//   class MicroMutableOpResolver;
// }

class MLInference {
public:
  MLInference();
  ~MLInference();
  
  // Initialize the TFLite interpreter
  bool begin();
  
  // Run inference on input data
  bool predict(float* input_data, float* output_data);
  
  // Get the predicted sleep stage
  SleepStage getPredictedStage(float* output_data);
  
  // Get model input size
  int getInputSize() const { return MODEL_INPUT_SIZE; }
  
  // Get model output size  
  int getOutputSize() const { return MODEL_OUTPUT_SIZE; }

private:
  void* interpreter_;  // Generic pointer for testing
  void* resolver_;     // Generic pointer for testing
  uint8_t* tensor_arena_;
  bool initialized_;
};