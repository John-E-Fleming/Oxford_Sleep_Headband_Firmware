#pragma once

// Replace this with your actual model data array
// Generated from: xxd -i model.tflite > model.h
// Then format as: const unsigned char model_tflite[] = { ... };

extern const unsigned char model_tflite[];
extern const int model_tflite_len;

// Model input/output specifications  
// Reduced for memory constraints on Teensy 4.1
#define MODEL_INPUT_SIZE (1000)      // 1000 samples for testing (reduced from 16000)
#define MODEL_OUTPUT_SIZE 4          // Adjust based on number of classes
#define MODEL_TENSOR_ARENA_SIZE (10 * 1024)  // 10KB arena (reduced from 100KB)

// Sleep stage classification labels (example - update for your model)
enum SleepStage {
  WAKE = 0,
  LIGHT_SLEEP = 1,
  DEEP_SLEEP = 2,
  REM_SLEEP = 3
};