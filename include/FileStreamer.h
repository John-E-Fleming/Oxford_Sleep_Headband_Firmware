#pragma once
#include <SdFat.h>
#include <Arduino.h>

// Initialize SD & root directory. Returns true on success.
bool initSD(const SdSpiConfig &sdConfig, SdFat &sdLib, SdFile &rootDir);

// Open `filename` from root, prepare for streaming. Returns true if file opened.
bool openDataFile(SdFile &rootDir, const String &filename, SdFile &dataFile);

// Call repeatedly in loop() to stream until EOF.
// Returns true while still streaming, false when done.
bool streamOnce(SdFile &dataFile);
