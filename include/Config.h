#pragma once
#include <SdFat.h>
#include <Arduino.h>

// Holds your key-values
struct Config {
  String datafile;
  int sample_rate;
  int channels;
  String format;
  int gain;
  float vref;
  int bipolar_channel_positive;
  int bipolar_channel_negative;
  int ml_target_sample_rate;
  int ml_window_seconds;
};

// Read and parse "config.txt" into a Config object.
// Returns true on success.
bool loadConfig(SdFile &cardRoot, Config &cfg);
