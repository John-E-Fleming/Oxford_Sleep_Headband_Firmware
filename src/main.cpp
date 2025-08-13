#include <Arduino.h>
#include <SPI.h>
#include <SdFat.h>          // Library for SD card access
#include <RingBuf.h>     // Library for ring buffer
#include "Config.h"
#include "FileStreamer.h"

// — SD setup —
SdFat sd;
SdFile rootDir;
SdFile dataFile;
Config cfg;

// SPI/SDIO configuration for Teensy 4.1
#define SD_CFG SdSpiConfig(10, DEDICATED_SPI, SD_SCK_MHZ(50), &SPI1)

static constexpr uint8_t SD_CS_PIN = 10;        // Teensy 4.1 built-in SD slot CS
static constexpr uint32_t SD_SPEED = SD_SCK_MHZ(50);

void setup() {
  Serial.begin(115200);
  while (!Serial) {}

  if (!initSD(SD_CFG, sd, rootDir)) {
    Serial.println(F("SD init failed")); while (1);
  }
  Serial.println(F("SD initialized"));

  if (!loadConfig(rootDir, cfg)) {
    Serial.println(F("Failed to load config.txt")); while (1);
  }
  Serial.print(F("Data file = "));
  Serial.println(cfg.datafile);

  if (!openDataFile(rootDir, cfg.datafile, dataFile)) {
    Serial.println(F("Could not open data file")); while (1);
  }
  Serial.println(F("Streaming…"));
}

void loop() {
  
  if (!streamOnce(dataFile)) {
    Serial.println(F("\nDone."));
    while (1);
  }
}
