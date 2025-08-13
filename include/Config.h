#pragma once
#include <SdFat.h>
#include <Arduino.h>

// Holds your key-values
struct Config {
  String datafile;
  // add more settings here as needed
};

// Read and parse "config.txt" into a Config object.
// Returns true on success.
bool loadConfig(SdFile &cardRoot, Config &cfg);
