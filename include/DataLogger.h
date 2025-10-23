#pragma once

#include <Arduino.h>
#include <SdFat.h>

// Logs raw and processed EEG data for offline analysis and comparison
// Writes binary files for efficient storage and fast writing
class DataLogger {
public:
  DataLogger();
  ~DataLogger();

  // Initialize logger with separate files for raw and normalized data
  bool begin(const String& session_name = "");

  // Log a single raw filtered sample (bipolar derivation after bandpass)
  void logRawSample(float sample);

  // Log an entire normalized window (3000 z-scored samples)
  void logNormalizedWindow(const float* window, int size, int epoch_index);

  // Flush data to SD card (call periodically to ensure data is saved)
  void flush();

  // Close files
  void close();

  // Check if logging is active
  bool isLogging() const { return raw_file_.isOpen() && normalized_file_.isOpen(); }

  // Get statistics
  unsigned long getRawSampleCount() const { return raw_sample_count_; }
  unsigned long getNormalizedWindowCount() const { return normalized_window_count_; }

private:
  SdFile raw_file_;              // Binary file for raw filtered samples
  SdFile normalized_file_;       // Binary file for normalized windows
  SdFile metadata_file_;         // Text file with metadata

  bool initialized_;
  unsigned long raw_sample_count_;
  unsigned long normalized_window_count_;

  String session_name_;

  // Write metadata about the data files
  void writeMetadata();
};
