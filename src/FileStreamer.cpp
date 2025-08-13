#include "FileStreamer.h"

// e.g. SdSpiConfig or SdioConfig instance passed in from main
bool initSD(const SdSpiConfig &sdConfig, SdFat &sdLib, SdFile &rootDir) {
  if (!sdLib.begin(sdConfig)) return false;
  if (!rootDir.open("/", O_READ)) return false;
  return true;
}

bool openDataFile(SdFile &rootDir, const String &filename, SdFile &dataFile) {
  return dataFile.open(filename.c_str(), O_READ);
}

bool streamOnce(SdFile &dataFile) {
  if (dataFile.available()) {
    uint8_t buf[64];
    size_t n = dataFile.read(buf, sizeof(buf));
    Serial.write(buf, n);
    return true;
  }
  dataFile.close();
  return false;
}
