#pragma once

#include <Arduino.h>
#include "model.h"

// TensorFlow Lite Micro includes - using Chirale TensorFlow Lite library
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

class MLInference {
public:
  MLInference();
  ~MLInference();

  // Initialize the TFLite interpreter
  bool begin(bool use_dummy = false);

  // Run inference on standardized float data with epoch index
  bool predict(float* input_data, float* output_data, int epoch_index);

  // Get the predicted sleep stage
  SleepStage getPredictedStage(float* output_data);

  // Get model input size
  int getInputSize() const { return MODEL_INPUT_SIZE; }

  // Get model output size
  int getOutputSize() const { return MODEL_OUTPUT_SIZE; }

  // Get inference timing (microseconds)
  unsigned long getLastInferenceTime() const { return last_inference_time_us_; }

  // Check if using real model or dummy
  bool isUsingRealModel() const { return !use_dummy_model_ && initialized_; }

  // Get quantization parameters for INT8 input
  bool getInputQuantizationParams(float& scale, int32_t& zero_point) const;

private:
  // TensorFlow Lite components
  const tflite::Model* model_;
  tflite::MicroInterpreter* interpreter_;
  TfLiteTensor* input_tensor_;
  TfLiteTensor* output_tensor_;

  // Memory management
  uint8_t* tensor_arena_;
  bool tensor_arena_is_extmem_;  // Track allocation method

  bool initialized_;
  bool use_dummy_model_;  // Flag for testing without real model
  unsigned long last_inference_time_us_;  // Timing measurement
};