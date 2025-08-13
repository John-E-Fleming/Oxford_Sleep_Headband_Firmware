#include "EEGFileReader.h"

EEGFileReader::EEGFileReader() 
  : format_(FORMAT_INT32), num_channels_(9), 
    bytes_per_channel_(4), current_position_(0) {
  bytes_per_sample_ = num_channels_ * bytes_per_channel_;
}

bool EEGFileReader::begin(const String& filename) {
  file_ = SD.open(filename.c_str(), FILE_READ);
  if (!file_) {
    Serial.print("Failed to open file: ");
    Serial.println(filename);
    return false;
  }
  
  file_size_ = file_.size();
  current_position_ = 0;
  
  Serial.print("Opened EEG file: ");
  Serial.println(filename);
  Serial.print("File size: ");
  Serial.print(file_size_);
  Serial.println(" bytes");
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
  if (!file_) {
    return false;
  }
  
  // Check if we have enough data left
  if (current_position_ + bytes_per_sample_ > file_size_) {
    return false; // End of file
  }
  
  uint8_t raw_data[32]; // Max 8 channels * 4 bytes = 32 bytes
  
  if (file_.read(raw_data, bytes_per_sample_) != bytes_per_sample_) {
    return false;
  }
  
  current_position_ += bytes_per_sample_;
  
  // Convert raw data to float values for each channel
  for (int ch = 0; ch < num_channels_; ch++) {
    channel_data[ch] = convertToFloat(raw_data, ch);
  }
  
  return true;
}

bool EEGFileReader::seekToTime(float seconds) {
  uint32_t sample_number = (uint32_t)(seconds * SAMPLE_RATE);
  uint32_t byte_position = sample_number * bytes_per_sample_;
  
  if (byte_position >= file_size_) {
    return false;
  }
  
  if (file_.seek(byte_position)) {
    current_position_ = byte_position;
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