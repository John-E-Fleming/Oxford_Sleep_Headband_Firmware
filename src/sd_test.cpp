#include <Arduino.h>
#include <SPI.h>
#include <SdFat.h>

// SdFat objects
SdFat sd;
SdFile root;
SdFile file;

// Forward declarations
void listDirectory(SdFile dir, int numTabs);
void testFileAccess(const char* filename);

// Memory usage function for Teensy
extern "C" char* sbrk(int incr);
int freeMemory() {
  char top;
  return &top - reinterpret_cast<char*>(sbrk(0));
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 10000) {} // Wait up to 10 seconds for serial
  
  Serial.println("=== Teensy 4.1 SD Card Debug Test ===");
  Serial.println();
  
  // Basic hardware checks
  Serial.println("=== Hardware Information ===");
  Serial.print("Teensy CPU Frequency: ");
  Serial.print(F_CPU / 1000000);
  Serial.println(" MHz");
  Serial.print("Free RAM: ");
  Serial.print(freeMemory());
  Serial.println(" bytes");
  Serial.println();
  
  // Test SD card initialization with different methods (using SdFat library)
  Serial.println("Testing SD card initialization methods...");
  
  bool sd_success = false;
  
  // Method 1: SdioConfig (like your colleague's code)
  Serial.print("1. Trying SdioConfig(FIFO_SDIO): ");
  if (sd.begin(SdioConfig(FIFO_SDIO))) {
    Serial.println("SUCCESS");
    sd_success = true;
  } else {
    Serial.println("FAILED");
  }
  
  if (!sd_success) {
    // Method 2: Basic SDIO
    Serial.print("2. Trying basic SdioConfig: ");
    if (sd.begin(SdioConfig(DMA_SDIO))) {
      Serial.println("SUCCESS");
      sd_success = true;
    } else {
      Serial.println("FAILED");
    }
  }
  
  if (!sd_success) {
    // Method 3: SPI with pin 10
    Serial.print("3. Trying SPI with CS pin 10: ");
    if (sd.begin(SdSpiConfig(10, DEDICATED_SPI, SD_SCK_MHZ(50)))) {
      Serial.println("SUCCESS");
      sd_success = true;
    } else {
      Serial.println("FAILED");
    }
  }
  
  if (!sd_success) {
    // Method 4: Default SPI
    Serial.print("4. Trying default SPI settings: ");
    if (sd.begin(SdSpiConfig(10, SHARED_SPI))) {
      Serial.println("SUCCESS");
      sd_success = true;
    } else {
      Serial.println("FAILED");
    }
  }

  if (!sd_success) {
    Serial.println("\nERROR: All SD initialization methods failed!");
    Serial.println("\n=== Troubleshooting Guide ===");
    Serial.println("1. Physical Check:");
    Serial.println("   - SD card fully inserted in Teensy 4.1 built-in slot?");
    Serial.println("   - SD card ≤32GB and formatted as FAT32?");
    Serial.println("   - Adequate power supply (USB may not be sufficient)?");
    Serial.println();
    Serial.println("2. Try These Steps:");
    Serial.println("   - Remove and reinsert SD card");
    Serial.println("   - Format SD card as FAT32 on computer");
    Serial.println("   - Try different SD card");
    Serial.println("   - Use external power supply instead of USB");
    Serial.println();
    Serial.println("3. Alternative Wiring (if using external SD module):");
    Serial.println("   - CS: pin 10");
    Serial.println("   - MOSI: pin 11");
    Serial.println("   - MISO: pin 12");
    Serial.println("   - SCK: pin 13");
    Serial.println("\nPress any key to retry initialization...");
    
    // Wait for user input to retry
    while (!Serial.available()) {
      delay(100);
    }
    while (Serial.available()) Serial.read(); // Clear buffer
    setup(); // Restart test
    return;
  }
  
  Serial.println("\n=== SD Card Information ===");
  Serial.println("SD card initialized successfully");
  Serial.println("(Card type and size info not available with this SD library)");
  
  // List all files on SD card
  Serial.println("\n=== Files on SD Card ===");
  if (!root.open("/")) {
    Serial.println("ERROR: Failed to open root directory");
  } else {
    listDirectory(root, 0);
    root.close();
  }
  
  // Test specific file access
  Serial.println("\n=== Testing File Access ===");
  testFileAccess("SdioLogger_miklos_night_2.bin");
  testFileAccess("config.txt");
  testFileAccess("data/example_datasets/random/random_numbers.txt");
  testFileAccess("random_numbers.txt"); // In case it's in root
  
  Serial.println("\n=== Test Complete ===");
}

void loop() {
  // Send keep-alive message every 10 seconds
  static unsigned long lastPrint = 0;
  if (millis() - lastPrint > 10000) {
    Serial.println("SD test running... (send any character to restart test)");
    lastPrint = millis();
  }
  
  // Restart test if user sends any character
  if (Serial.available()) {
    while (Serial.available()) Serial.read(); // Clear buffer
    Serial.println("\nRestarting test...");
    delay(1000);
    setup();
  }
}

void listDirectory(SdFile dir, int numTabs) {
  while (true) {
    SdFile entry;
    if (!entry.openNext(&dir, O_RDONLY)) {
      break;
    }
    
    for (uint8_t i = 0; i < numTabs; i++) {
      Serial.print('\t');
    }
    
    char name[64];
    entry.getName(name, sizeof(name));
    Serial.print(name);
    
    if (entry.isDir()) {
      Serial.println("/");
      listDirectory(entry, numTabs + 1);
    } else {
      Serial.print("\t\t");
      Serial.print(entry.fileSize());
      Serial.println(" bytes");
    }
    entry.close();
  }
}

void testFileAccess(const char* filename) {
  Serial.print("Testing file: ");
  Serial.println(filename);
  
  if (!sd.exists(filename)) {
    Serial.println("  - File does not exist");
    return;
  }
  
  if (!file.open(filename, O_RDONLY)) {
    Serial.println("  - File exists but cannot be opened");
    return;
  }
  
  Serial.print("  - File size: ");
  Serial.print(file.fileSize());
  Serial.println(" bytes");
  
  // Try to read first few bytes
  Serial.print("  - First 16 bytes (hex): ");
  for (int i = 0; i < 16 && file.available(); i++) {
    uint8_t b = file.read();
    if (b < 16) Serial.print("0");
    Serial.print(b, HEX);
    Serial.print(" ");
  }
  Serial.println();
  
  file.close();
  Serial.println("  - File access successful");
}