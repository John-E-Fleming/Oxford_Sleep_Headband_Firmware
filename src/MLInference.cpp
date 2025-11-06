#include "MLInference.h"

// Static objects for TFLite (must persist for lifetime of interpreter)
namespace {
  tflite::MicroMutableOpResolver<13> micro_op_resolver;
}

MLInference::MLInference()
  : model_(nullptr), interpreter_(nullptr),
    input_tensor_(nullptr), input_tensor_epoch_(nullptr), output_tensor_(nullptr),
    tensor_arena_(nullptr), tensor_arena_is_extmem_(false),
    initialized_(false), use_dummy_model_(false), last_inference_time_us_(0) {
}

MLInference::~MLInference() {
  if (interpreter_) {
    delete interpreter_;
  }
  if (tensor_arena_) {
    if (tensor_arena_is_extmem_) {
      extmem_free(tensor_arena_);
    } else {
      delete[] tensor_arena_;
    }
  }
}

bool MLInference::begin(bool use_dummy) {
  if (initialized_) {
    return true;
  }

  use_dummy_model_ = use_dummy;

  if (use_dummy_model_) {
    Serial.println("ML Inference: Using dummy model for testing");
    initialized_ = true;
    return true;
  }

  Serial.println("ML Inference: Initializing TensorFlow Lite Micro...");

  // Load the model
  extern const unsigned char model_tflite[];
  extern const int model_tflite_len;

  if (model_tflite_len == 0) {
    Serial.println("No embedded model found, falling back to dummy mode");
    use_dummy_model_ = true;
    initialized_ = true;
    return true;
  }

  Serial.print("Model size: ");
  Serial.print(model_tflite_len);
  Serial.println(" bytes");

  model_ = tflite::GetModel(model_tflite);
  if (model_->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model schema version ");
    Serial.print(model_->version());
    Serial.print(" not supported. Supported version is ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    return false;
  }

  // Add operations needed by the model (from reference implementation)
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddQuantize();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddMean();
  micro_op_resolver.AddAdd();
  micro_op_resolver.AddDequantize();
  micro_op_resolver.AddRelu();
  micro_op_resolver.AddConcatenation();

  // Allocate tensor arena
  tensor_arena_ = (uint8_t*)extmem_malloc(MODEL_TENSOR_ARENA_SIZE);
  if (tensor_arena_) {
    tensor_arena_is_extmem_ = true;
    Serial.println("Tensor arena allocated in external RAM");
  } else {
    Serial.println("Failed to allocate tensor arena in external RAM, trying regular RAM");
    tensor_arena_ = new uint8_t[MODEL_TENSOR_ARENA_SIZE];
    tensor_arena_is_extmem_ = false;
    if (!tensor_arena_) {
      Serial.println("Failed to allocate tensor arena");
      return false;
    }
    Serial.println("Tensor arena allocated in regular RAM");
  }

  // Build interpreter
  interpreter_ = new tflite::MicroInterpreter(
    model_, micro_op_resolver, tensor_arena_, MODEL_TENSOR_ARENA_SIZE, nullptr);

  if (!interpreter_) {
    Serial.println("Failed to create interpreter");
    return false;
  }

  // Allocate tensors
  TfLiteStatus allocate_status = interpreter_->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return false;
  }

  // Get input and output tensors (model has 2 inputs: EEG data + epoch index)
  input_tensor_ = interpreter_->input(0);        // EEG data tensor
  input_tensor_epoch_ = interpreter_->input(1);  // Epoch index tensor
  output_tensor_ = interpreter_->output(0);

  // Store tensor types for runtime handling
  input_type_ = input_tensor_->type;
  input_epoch_type_ = input_tensor_epoch_->type;
  output_type_ = output_tensor_->type;

  // Helper function to get type name
  auto getTypeName = [](TfLiteType type) -> const char* {
    switch (type) {
      case kTfLiteFloat32: return "FLOAT32";
      case kTfLiteInt8: return "INT8";
      case kTfLiteUInt8: return "UINT8";
      case kTfLiteInt32: return "INT32";
      default: return "UNKNOWN";
    }
  };

  // Helper function to get type size
  auto getTypeSize = [](TfLiteType type) -> int {
    switch (type) {
      case kTfLiteFloat32: return 4;
      case kTfLiteInt32: return 4;
      case kTfLiteInt8: return 1;
      case kTfLiteUInt8: return 1;
      default: return 1;
    }
  };

  // Print input tensor 0 (EEG data) info
  int type_size_0 = getTypeSize(input_type_);
  int input_eeg_elements = input_tensor_->bytes / type_size_0;
  Serial.print("Input tensor 0 (EEG): ");
  Serial.print(input_eeg_elements);
  Serial.print(" elements (");
  Serial.print(getTypeName(input_type_));
  Serial.print("), shape: ");
  for (int i = 0; i < input_tensor_->dims->size; i++) {
    if (i > 0) Serial.print("x");
    Serial.print(input_tensor_->dims->data[i]);
  }
  Serial.println();

  if (input_eeg_elements != MODEL_EEG_SAMPLES) {
    Serial.print("WARNING: Expected ");
    Serial.print(MODEL_EEG_SAMPLES);
    Serial.print(" EEG samples but model has ");
    Serial.println(input_eeg_elements);
  }

  // Print input tensor 1 (Epoch index) info
  int type_size_1 = getTypeSize(input_epoch_type_);
  int input_epoch_elements = input_tensor_epoch_->bytes / type_size_1;
  Serial.print("Input tensor 1 (Epoch): ");
  Serial.print(input_epoch_elements);
  Serial.print(" elements (");
  Serial.print(getTypeName(input_epoch_type_));
  Serial.print("), shape: ");
  for (int i = 0; i < input_tensor_epoch_->dims->size; i++) {
    if (i > 0) Serial.print("x");
    Serial.print(input_tensor_epoch_->dims->data[i]);
  }
  Serial.println();

  // Print output tensor info
  int type_size_out = getTypeSize(output_type_);
  int output_elements = output_tensor_->bytes / type_size_out;
  Serial.print("Output tensor: ");
  Serial.print(output_elements);
  Serial.print(" elements (");
  Serial.print(getTypeName(output_type_));
  Serial.println(")");

  if (output_elements != MODEL_OUTPUT_SIZE) {
    Serial.print("WARNING: Expected ");
    Serial.print(MODEL_OUTPUT_SIZE);
    Serial.print(" outputs but model has ");
    Serial.println(output_elements);
  }

  initialized_ = true;
  Serial.println("TensorFlow Lite Micro initialized successfully");
  return true;
}

bool MLInference::predict(float* input_data, float* output_data, int epoch_index) {
  if (!initialized_) {
    Serial.println("Model not initialized");
    return false;
  }

  unsigned long start_time = micros();

  if (use_dummy_model_) {
    // Return dummy prediction for testing
    float sum = 0.0f;
    for (int i = 0; i < 100; i++) {
      sum += input_data[i];
    }
    float mean_sample = sum / 100.0f;

    // Create somewhat realistic sleep stage probabilities
    // Correct mapping: [0]=Wake, [1]=N1, [2]=N2, [3]=N3, [4]=REM
    if (mean_sample > 1.0f) {
      output_data[0] = 0.8f;  // WAKE
      output_data[1] = 0.1f;  // N1
      output_data[2] = 0.05f; // N2
      output_data[3] = 0.03f; // N3
      output_data[4] = 0.02f; // REM
    } else if (mean_sample < -1.0f) {
      output_data[0] = 0.1f;  // WAKE
      output_data[1] = 0.1f;  // N1
      output_data[2] = 0.3f;  // N2
      output_data[3] = 0.4f;  // N3
      output_data[4] = 0.1f;  // REM
    } else {
      output_data[0] = 0.2f;  // WAKE
      output_data[1] = 0.3f;  // N1
      output_data[2] = 0.25f; // N2
      output_data[3] = 0.1f;  // N3
      output_data[4] = 0.15f; // REM
    }

    last_inference_time_us_ = micros() - start_time;
    return true;
  }

  // Real model inference - dual input architecture
  // Input data is already z-score normalized by EEGProcessor

  // Populate Input Tensor 0: EEG samples (shape: 1, 1, 3000, 1)
  if (input_type_ == kTfLiteFloat32) {
    // FLOAT32 model - copy data directly
    for (int i = 0; i < MODEL_EEG_SAMPLES; i++) {
      input_tensor_->data.f[i] = input_data[i];
    }
  } else if (input_type_ == kTfLiteInt8) {
    // INT8 quantized model - quantize data
    for (int i = 0; i < MODEL_EEG_SAMPLES; i++) {
      int8_t x_quantized = input_data[i] / input_tensor_->params.scale + input_tensor_->params.zero_point;
      input_tensor_->data.int8[i] = x_quantized;
    }
  } else {
    Serial.println("ERROR: Unsupported input tensor type");
    return false;
  }

  // Populate Input Tensor 1: Epoch index (shape: 1, 1)
  float f_epoch = (float)epoch_index;
  if (input_epoch_type_ == kTfLiteFloat32) {
    // FLOAT32 - copy directly
    input_tensor_epoch_->data.f[0] = f_epoch;
  } else if (input_epoch_type_ == kTfLiteInt8) {
    // INT8 - quantize
    int8_t epoch_quantized = f_epoch / input_tensor_epoch_->params.scale + input_tensor_epoch_->params.zero_point;
    input_tensor_epoch_->data.int8[0] = epoch_quantized;
  } else {
    Serial.println("ERROR: Unsupported epoch tensor type");
    return false;
  }

  // Run inference
  TfLiteStatus invoke_status = interpreter_->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed");
    return false;
  }

  // Extract output tensor (5 outputs)
  if (output_type_ == kTfLiteFloat32) {
    // FLOAT32 output - copy directly
    for (int i = 0; i < MODEL_OUTPUT_SIZE; i++) {
      output_data[i] = output_tensor_->data.f[i];
    }
  } else if (output_type_ == kTfLiteInt8) {
    // INT8 output - dequantize
    for (int i = 0; i < MODEL_OUTPUT_SIZE; i++) {
      int8_t y_quantized = output_tensor_->data.int8[i];
      output_data[i] = (y_quantized - output_tensor_->params.zero_point) * output_tensor_->params.scale;
    }
  } else {
    Serial.println("ERROR: Unsupported output tensor type");
    return false;
  }

  // Debug output for first few inferences
  static int debug_count = 0;
  if (debug_count < 3) {
    Serial.println("=== Model Output (dequantized) ===");
    for (int i = 0; i < MODEL_OUTPUT_SIZE; i++) {
      Serial.print("Output[");
      Serial.print(i);
      Serial.print("] = ");
      Serial.println(output_data[i], 6);
    }
    debug_count++;
  }

  last_inference_time_us_ = micros() - start_time;
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

bool MLInference::getInputQuantizationParams(float& scale, int32_t& zero_point) const {
  if (!initialized_ || use_dummy_model_) {
    // Return default quantization parameters for dummy model
    scale = 1.0f;
    zero_point = 0;
    return false;
  }

  if (!input_tensor_) {
    return false;
  }

  scale = input_tensor_->params.scale;
  zero_point = input_tensor_->params.zero_point;
  return true;
}
