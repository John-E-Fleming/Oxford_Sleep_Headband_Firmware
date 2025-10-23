#pragma once

// Replace this with your actual model data array
// Generated from: xxd -i model.tflite > model.h
// Then format as: const unsigned char model_tflite[] = { ... };

extern const unsigned char model_tflite[];
extern const int model_tflite_len;

// Model input/output specifications
// Matching the reference implementation: 30 seconds at 100Hz + epoch index
#define MODEL_INPUT_SIZE (3001)      // 3000 EEG samples + 1 epoch index
#define MODEL_EEG_SAMPLES (3000)     // 30 seconds at 100Hz = 3000 samples
#define MODEL_OUTPUT_SIZE 5          // 5 classes: Wake, N1, N2, N3, REM (matching reference)
#define MODEL_TENSOR_ARENA_SIZE (160 * 1024)   // 160KB arena to accommodate model needs

// Sleep stage classification labels - CORRECTED MAPPING from reference firmware
// Model output indices correspond to:
// Index 0 (yy0) = N3 (Deep Sleep)
// Index 1 (yy1) = N2 (Light Sleep) 
// Index 2 (yy2) = N1 (Very Light Sleep)
// Index 3 (yy3) = REM Sleep
// Index 4 (yy4) = Wake
enum SleepStage {
  N3_DEEP_SLEEP = 0,    // yy0 - Deep sleep (N3)
  N2_LIGHT_SLEEP = 1,   // yy1 - Light sleep (N2)
  N1_VERY_LIGHT = 2,    // yy2 - Very light sleep (N1)
  REM_SLEEP = 3,        // yy3 - REM sleep
  WAKE = 4              // yy4 - Wake
};