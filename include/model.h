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

// Sleep stage classification labels - CORRECT MAPPING from reference firmware
// Model output indices correspond to (verified from comp_comp_firm 1.ino:1158-1172):
// Index 0 (yy0) = Wake
// Index 1 (yy1) = N1 (Very Light Sleep)
// Index 2 (yy2) = N2 (Light Sleep)
// Index 3 (yy3) = N3 (Deep Sleep)
// Index 4 (yy4) = REM Sleep
enum SleepStage {
  WAKE = 0,             // yy0 - Wake
  N1_VERY_LIGHT = 1,    // yy1 - Very light sleep (N1)
  N2_LIGHT_SLEEP = 2,   // yy2 - Light sleep (N2)
  N3_DEEP_SLEEP = 3,    // yy3 - Deep sleep (N3)
  REM_SLEEP = 4         // yy4 - REM sleep
};