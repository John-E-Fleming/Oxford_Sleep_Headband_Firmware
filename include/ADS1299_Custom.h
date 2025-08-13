#pragma once

#include <Arduino.h>
#include <SPI.h>

// ADS1299 Pin definitions (adjust based on your hardware)
#define ADS1299_CS1   7    // Chip Select
#define ADS1299_DRDY  22   // Data Ready (interrupt pin)
#define ADS1299_START_PIN 15   // Start conversion pin
#define ADS1299_PWDN  14   // Power down
#define ADS1299_CHANNELS 8 // Number of EEG channels

// ADS1299 Register addresses
#define CONFIG1_REG   0x01
#define CONFIG2_REG   0x02  
#define CONFIG3_REG   0x03
#define CH1SET_REG    0x05
#define CH2SET_REG    0x06
#define CH3SET_REG    0x07
#define CH4SET_REG    0x08
#define CH5SET_REG    0x09
#define CH6SET_REG    0x0A
#define CH7SET_REG    0x0B
#define CH8SET_REG    0x0C

// ADS1299 Commands
#define ADS1299_WAKEUP  0x02
#define ADS1299_STANDBY 0x04
#define ADS1299_RESET   0x06
#define ADS1299_START_CMD   0x08  // Renamed to avoid conflict
#define ADS1299_STOP    0x0A
#define ADS1299_RDATAC  0x10
#define ADS1299_SDATAC  0x11
#define ADS1299_RDATA   0x12

// Gain settings
#define GAIN_1X   0x00  // step: 536nV, range: +/- 4.500V
#define GAIN_2X   0x10  // step: 268nV, range: +/- 2.250V
#define GAIN_4X   0x20  // step: 134nV, range: +/- 1.125V
#define GAIN_6X   0x30  // step:  89nV, range: +/- 0.750V
#define GAIN_8X   0x40  // step:  67nV, range: +/- 0.563V
#define GAIN_12X  0x50  // step:  45nV, range: +/- 0.375V
#define GAIN_24X  0x60  // step:  22nV, range: +/- 0.188V

// Sampling rates
#define FS_16KHZ  0x00
#define FS_8KHZ   0x01
#define FS_4KHZ   0x02
#define FS_2KHZ   0x03
#define FS_1KHZ   0x04
#define FS_500HZ  0x05
#define FS_250HZ  0x06

// Channel settings
#define CHSET_INPUT 0x00  // Normal electrode input
#define CHSET_SHORT 0x01  // Input shorted
#define CHSET_TEST  0x05  // Test signal input

class ADS1299_Custom {
public:
  ADS1299_Custom();
  
  // Initialize ADS1299
  bool begin();
  
  // Start/stop data acquisition
  void startAcquisition();
  void stopAcquisition();
  
  // Check if new data is available
  bool dataReady();
  
  // Read channel data (blocking)
  void readChannelData(int32_t* channelData);
  
  // Read channel data and convert to microvolts
  void readChannelDataVolts(float* channelData);
  
  // Configuration functions
  void setGain(uint8_t gain);
  void setSampleRate(uint8_t sampleRate);
  void configureChannels();
  
  // Low-level register access
  void writeRegister(uint8_t reg, uint8_t value);
  uint8_t readRegister(uint8_t reg);
  
  // Interrupt service routine (call from attachInterrupt)
  static void dataReadyISR();
  
  // Get data from ISR buffer
  bool getLatestData(int32_t* channelData);
  
  // Convert raw ADC value to microvolts
  float convertToMicrovolts(int32_t adcValue);

private:
  SPISettings spiSettings_;
  bool initialized_;
  uint8_t currentGain_;
  float gainMultiplier_;
  
  // ISR data buffer
  static volatile int32_t isrBuffer_[ADS1299_CHANNELS];
  static volatile bool newDataAvailable_;
  
  // Helper functions
  void reset();
  void powerUp();
  void sendCommand(uint8_t command);
  void updateGainMultiplier();
};