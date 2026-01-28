#include "EEGFileReader.h"

// Global SdFat object (will be defined in main files)
extern SdFat sd;

EEGFileReader::EEGFileReader()
  : format_(FORMAT_INT32), num_channels_(9),
    bytes_per_channel_(4), current_position_(0),
    buffer_position_(0), buffer_valid_bytes_(0),
    buffering_enabled_(true) {
  bytes_per_sample_ = num_channels_ * bytes_per_channel_;

  // Allocate buffer in external RAM (EXTMEM) for optimal performance
  read_buffer_ = (uint8_t*)extmem_malloc(BUFFER_SIZE);
  if (!read_buffer_) {
    Serial.println("WARNING: Failed to allocate read buffer in EXTMEM, trying regular malloc");
    read_buffer_ = (uint8_t*)malloc(BUFFER_SIZE);
    if (!read_buffer_) {
      Serial.println("ERROR: Failed to allocate read buffer! Buffering disabled.");
      buffering_enabled_ = false;
    } else {
      Serial.println("EEG buffered reading: ENABLED (regular malloc)");
    }
  } else {
    Serial.println("EEG buffered reading: ENABLED (EXTMEM)");
  }
}

EEGFileReader::~EEGFileReader() {
  close();
  if (read_buffer_) {
    free(read_buffer_);  // Works for both extmem_malloc and malloc
    read_buffer_ = nullptr;
  }
}

bool EEGFileReader::begin(const String& filename) {
  // First check if file exists
  if (!sd.exists(filename.c_str())) {
    Serial.print("ERROR: File does not exist: ");
    Serial.println(filename);
    Serial.println("Available files on SD card:");
    SdFile root;
    if (root.open("/")) {
      while (true) {
        SdFile entry;
        if (!entry.openNext(&root, O_RDONLY)) break;
        char name[64];
        entry.getName(name, sizeof(name));
        Serial.print("  ");
        Serial.print(name);
        Serial.print(" (");
        Serial.print(entry.fileSize());
        Serial.println(" bytes)");
        entry.close();
      }
      root.close();
    } else {
      Serial.println("ERROR: Cannot open root directory");
    }
    return false;
  }
  
  // Try to open the file
  if (!file_.open(filename.c_str(), O_RDONLY)) {
    Serial.print("ERROR: Failed to open file: ");
    Serial.println(filename);
    Serial.println("File exists but cannot be opened - check file permissions or corruption");
    return false;
  }
  
  file_size_ = file_.fileSize();
  current_position_ = 0;
  
  // Validate file size makes sense for our format
  if (file_size_ == 0) {
    Serial.println("ERROR: File is empty");
    file_.close();
    return false;
  }
  
  // Check if file size is reasonable for EEG data
  uint32_t expected_samples = file_size_ / bytes_per_sample_;
  if (expected_samples == 0) {
    Serial.println("ERROR: File too small for expected format");
    Serial.print("File size: ");
    Serial.print(file_size_);
    Serial.print(" bytes, Expected bytes per sample: ");
    Serial.println(bytes_per_sample_);
    file_.close();
    return false;
  }
  
  Serial.print("SUCCESS: Opened EEG file: ");
  Serial.println(filename);
  Serial.print("File size: ");
  Serial.print(file_size_);
  Serial.println(" bytes");
  Serial.print("Channels: ");
  Serial.print(num_channels_);
  Serial.print(", Bytes per sample: ");
  Serial.println(bytes_per_sample_);
  Serial.print("Total samples: ");
  Serial.println(expected_samples);
  Serial.print("Estimated duration: ");
  Serial.print(getDurationSeconds(), 1);
  Serial.println(" seconds");
  
  return true;
}

void EEGFileReader::setFormat(EEGDataFormat format, int channels) {
  format_ = format;
  num_channels_ = channels;
  
  switch (format_) {
    case FORMAT_ADS1299_24BIT:
      bytes_per_channel_ = 3;
      break;
    case FORMAT_FLOAT32:
    case FORMAT_INT32:
      bytes_per_channel_ = 4;
      break;
    case FORMAT_INT16:
      bytes_per_channel_ = 2;
      break;
  }
  
  bytes_per_sample_ = num_channels_ * bytes_per_channel_;
}

bool EEGFileReader::readNextSample(float* channel_data) {
  // Use buffered reading if enabled, otherwise fall back to direct reading
  if (!buffering_enabled_ || !read_buffer_) {
    return readNextSampleDirect(channel_data);
  }

  if (!file_) {
    Serial.println("ERROR: File not open");
    return false;
  }

  // Check if we have enough data left in file
  if (current_position_ + bytes_per_sample_ > file_size_) {
    return false; // End of file
  }

  // Refill buffer if needed
  if (buffer_position_ + bytes_per_sample_ > buffer_valid_bytes_) {
    if (!refillBuffer()) {
      return false;  // EOF or read error
    }
  }

  // Read from buffer instead of file
  uint8_t raw_data[64];
  memcpy(raw_data, read_buffer_ + buffer_position_, bytes_per_sample_);
  buffer_position_ += bytes_per_sample_;
  current_position_ += bytes_per_sample_;

  // Convert raw data to float values for each channel
  for (int ch = 0; ch < num_channels_; ch++) {
    channel_data[ch] = convertToFloat(raw_data, ch);
  }

  return true;
}

bool EEGFileReader::readNextSampleDirect(float* channel_data) {
  // Original unbuffered implementation
  if (!file_) {
    Serial.println("ERROR: File not open in readNextSampleDirect");
    return false;
  }

  // Check if we have enough data left
  if (current_position_ + bytes_per_sample_ > file_size_) {
    return false; // End of file
  }

  uint8_t raw_data[64]; // Buffer for 9 channels * 4 bytes = 36 bytes + margin

  int bytes_read = file_.read(raw_data, bytes_per_sample_);
  if (bytes_read != (int)bytes_per_sample_) {
    Serial.print("ERROR: Expected ");
    Serial.print(bytes_per_sample_);
    Serial.print(" bytes, but read ");
    Serial.print(bytes_read);
    Serial.println(" bytes");
    return false;
  }

  current_position_ += bytes_per_sample_;

  // Convert raw data to float values for each channel
  for (int ch = 0; ch < num_channels_; ch++) {
    channel_data[ch] = convertToFloat(raw_data, ch);
  }

  return true;
}

bool EEGFileReader::refillBuffer() {
  // Calculate how many bytes remain in the file
  uint32_t bytes_remaining = file_size_ - current_position_;

  if (bytes_remaining == 0) {
    return false;  // End of file
  }

  // CRITICAL FIX: Seek to current_position_ before reading to handle buffer alignment
  // The file cursor may be ahead of current_position_ if the previous buffer had
  // leftover bytes that weren't a complete sample.
  if (!file_.seek(current_position_)) {
    Serial.println("ERROR: Failed to seek in refillBuffer");
    return false;
  }

  // Read as much as possible (up to BUFFER_SIZE)
  int bytes_to_read = min((uint32_t)BUFFER_SIZE, bytes_remaining);

  // Debug: Print first few buffer refills
  static int refill_count = 0;
  if (refill_count < 5) {
    Serial.print("[BUFFER] Refilling buffer #");
    Serial.print(refill_count);
    Serial.print(" at position ");
    Serial.print(current_position_);
    Serial.print(" - reading ");
    Serial.print(bytes_to_read);
    Serial.println(" bytes");
    refill_count++;
  }

  int bytes_read = file_.read(read_buffer_, bytes_to_read);

  if (bytes_read <= 0) {
    Serial.println("ERROR: Failed to refill buffer from SD card");
    return false;
  }

  buffer_valid_bytes_ = bytes_read;
  buffer_position_ = 0;

  return true;
}

void EEGFileReader::enableBuffering(bool enable) {
  buffering_enabled_ = enable;

  // Reset buffer state when toggling
  buffer_position_ = 0;
  buffer_valid_bytes_ = 0;

  Serial.print("EEG buffered reading: ");
  Serial.println(enable ? "ENABLED" : "DISABLED");
}

bool EEGFileReader::seekToTime(float seconds) {
  uint32_t sample_number = (uint32_t)(seconds * SAMPLE_RATE);
  uint32_t byte_position = sample_number * bytes_per_sample_;

  if (byte_position >= file_size_) {
    return false;
  }

  if (file_.seek(byte_position)) {
    current_position_ = byte_position;

    // Invalidate buffer after seek
    buffer_position_ = 0;
    buffer_valid_bytes_ = 0;

    return true;
  }

  return false;
}

uint32_t EEGFileReader::getTotalSamples() const {
  return file_size_ / bytes_per_sample_;
}

float EEGFileReader::getDurationSeconds() const {
  return (float)getTotalSamples() / SAMPLE_RATE;
}

void EEGFileReader::close() {
  if (file_) {
    file_.close();
  }
  current_position_ = 0;
  file_size_ = 0;

  // Reset buffer state
  buffer_position_ = 0;
  buffer_valid_bytes_ = 0;
}

float EEGFileReader::convertToFloat(uint8_t* raw_data, int channel) {
  uint8_t* channel_data = raw_data + (channel * bytes_per_channel_);
  
  switch (format_) {
    case FORMAT_ADS1299_24BIT: {
      int32_t value = read24BitSigned(channel_data);
      // Convert to microvolts (typical ADS1299 scaling)
      // Adjust this scaling based on your ADC settings
      float voltage = (float)value * 4.5f / (1 << 23) * 1000000.0f; // µV
      return voltage;
    }
    
    case FORMAT_FLOAT32: {
      float value;
      memcpy(&value, channel_data, 4);
      return value;
    }
    
    case FORMAT_INT32: {
      int32_t value;
      memcpy(&value, channel_data, 4);
      // Convert ADC counts to microvolts using MATLAB formula
      // MATLAB: x * ((2.*4.5)./gain)./(2.^24).*1e6
      // gain = 24, reference voltage = 4.5V
      const float gain = 24.0f;
      const float vref = 4.5f;
      float voltage_uV = (float)value * (2.0f * vref / gain) / (1UL << 24) * 1000000.0f;
      return voltage_uV;
    }
    
    case FORMAT_INT16: {
      int16_t value;
      memcpy(&value, channel_data, 2);
      return (float)value;
    }
  }
  
  return 0.0f;
}

int32_t EEGFileReader::read24BitSigned(uint8_t* data) {
  // Assume little-endian format (adjust if needed)
  int32_t value = (data[2] << 16) | (data[1] << 8) | data[0];
  
  // Sign extend from 24-bit to 32-bit
  if (value & 0x800000) {
    value |= 0xFF000000;
  }
  
  return value;
}