#pragma once

#include <Arduino.h>
#include <SD.h>

// Common EEG data formats
enum EEGDataFormat {
  FORMAT_ADS1299_24BIT,  // 24-bit signed integers from ADS1299
  FORMAT_FLOAT32,        // 32-bit float values
  FORMAT_INT32,          // 32-bit signed integers
  FORMAT_INT16           // 16-bit signed integers
};

class EEGFileReader {
public:
  EEGFileReader();
  
  // Initialize with file
  bool begin(const String& filename);
  
  // Set data format parameters
  void setFormat(EEGDataFormat format, int channels = 8);
  
  // Read next sample (all channels)
  bool readNextSample(float* channel_data);
  
  // Skip to specific time offset (seconds)
  bool seekToTime(float seconds);
  
  // Get file info
  uint32_t getFileSize() const { return file_size_; }
  uint32_t getTotalSamples() const;
  float getDurationSeconds() const;
  bool isOpen() const { return (bool)file_; }
  
  // Close file
  void close();

private:
  File file_;
  EEGDataFormat format_;
  int num_channels_;
  int bytes_per_sample_;
  int bytes_per_channel_;
  uint32_t file_size_;
  uint32_t current_position_;
  
  // Sample rate from MATLAB analysis  
  static const int SAMPLE_RATE = 4000; // Hz
  
  // Helper functions
  float convertToFloat(uint8_t* raw_data, int channel);
  int32_t read24BitSigned(uint8_t* data);
};