#include <Arduino.h>
#include <SPI.h>
#include <SdFat.h>          // Library for SD card access
#include <RingBuf.h>     // Library for ring buffer

// put function declarations here:
int myFunction(int, int);

// — SD setup —
SdFat sd;
SdFile file;

// Tell SdFat to use the SDIO interface
#define SD_CONFIG SdioConfig(FIFO_SDIO)

static constexpr uint8_t SD_CS_PIN = 10;        // Teensy 4.1 built-in SD slot CS
static constexpr uint32_t SD_SPEED = SD_SCK_MHZ(50);

void setup() {
  Serial.begin(115200);
  while (!Serial) {}    // wait for USB Serial

  if (!sd.begin(SD_CONFIG)) {
    Serial.println(F("SD init failed"));
    while (1) {}
  }
  Serial.println(F("SD initialized at 50 MHz!"));;

  if (!file.open("random_numbers.txt", O_READ)) {
    Serial.println(F("❌ File open failed"));
    while (1) {}
  }

  Serial.println(F("📂 Streaming file contents to Serial…"));
}

void loop() {
  
  // As long as there's data on the SD card, read it and send it out immediately.
  if (file.available()) {
    // Read up to 64 bytes at once into a small buffer
    uint8_t buf[64];
    size_t n = file.read(buf, sizeof(buf));
    // Write those bytes directly to Serial (binary‐safe)
    Serial.write(buf, n);
  }
  else {
    // No more data → close file and stop
    file.close();
    Serial.println(F("\n✅ Done streaming."));
    while (1) {}
  }
  
}

// put function definitions here:
int myFunction(int x, int y) {
  return x + y;
}
