#pragma once

#include <Arduino.h>
#include <SdFat.h>

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
  ~EEGFileReader();

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
  bool isOpen() const { return file_size_ > 0; }

  // Buffered reading control
  void enableBuffering(bool enable = true);

  // Close file
  void close();

private:
  SdFile file_;
  EEGDataFormat format_;
  int num_channels_;
  int bytes_per_sample_;
  int bytes_per_channel_;
  uint32_t file_size_;
  uint32_t current_position_;

  // Sample rate from MATLAB analysis
  static const int SAMPLE_RATE = 4000; // Hz

  // Buffering system for fast SD card reads
  static const int BUFFER_SIZE = 16384;  // 16KB buffer = ~455 samples
  uint8_t* read_buffer_;                  // Buffer allocated in EXTMEM
  int buffer_position_;                   // Current read position in buffer (bytes)
  int buffer_valid_bytes_;                // Number of valid bytes in buffer
  bool buffering_enabled_;                // Enable/disable buffering

  // Helper functions
  float convertToFloat(uint8_t* raw_data, int channel);
  int32_t read24BitSigned(uint8_t* data);
  bool refillBuffer();                    // Refill buffer from SD card
  bool readNextSampleDirect(float* channel_data);  // Unbuffered reading
};