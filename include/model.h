#pragma once

// Replace this with your actual model data array
// Generated from: xxd -i model.tflite > model.h
// Then format as: const unsigned char model_tflite[] = { ... };

extern const unsigned char model_tflite[];
extern const int model_tflite_len;

// Model input/output specifications  
// Matching the reference implementation: 30 seconds at 100Hz + epoch index
#define MODEL_INPUT_SIZE (3001)      // 3000 samples + 1 epoch index (matching reference)
#define MODEL_OUTPUT_SIZE 5          // 5 classes: Wake, N1, N2, N3, REM (matching reference)
#define MODEL_TENSOR_ARENA_SIZE (150 * 1024)  // 150KB arena (matching reference kTensorArenaSize1)

// Sleep stage classification labels (example - update for your model)
enum SleepStage {
  WAKE = 0,
  LIGHT_SLEEP = 1,
  DEEP_SLEEP = 2,
  REM_SLEEP = 3
};