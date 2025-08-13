#include "ADS1299_Custom.h"

// Static variables for ISR
volatile int32_t ADS1299_Custom::isrBuffer_[ADS1299_CHANNELS] = {0};
volatile bool ADS1299_Custom::newDataAvailable_ = false;

ADS1299_Custom::ADS1299_Custom() 
  : spiSettings_(20E6, MSBFIRST, SPI_MODE1),
    initialized_(false),
    currentGain_(GAIN_24X),
    gainMultiplier_(0.0) {
}

bool ADS1299_Custom::begin() {
  // Initialize SPI
  SPI.begin();
  
  // Configure pins
  pinMode(ADS1299_CS1, OUTPUT);
  pinMode(ADS1299_START_PIN, OUTPUT);
  pinMode(ADS1299_PWDN, OUTPUT);
  pinMode(ADS1299_DRDY, INPUT);
  
  // Initial pin states
  digitalWriteFast(ADS1299_CS1, HIGH);
  digitalWriteFast(ADS1299_START_PIN, LOW);
  digitalWriteFast(ADS1299_PWDN, LOW);
  
  // Power up sequence
  powerUp();
  
  // Device configuration
  reset();
  delay(10); // Wait for reset
  
  // Stop data read command
  sendCommand(ADS1299_SDATAC);
  
  // Configure for 4kHz sampling (matching MATLAB analysis)
  writeRegister(CONFIG1_REG, 0x90 | FS_4KHZ); // Clock output enabled, 4kHz
  writeRegister(CONFIG3_REG, 0xE0); // Internal reference enabled, bias off
  
  delay(10); // Wait for reference to settle
  
  // Read device ID for verification
  uint8_t deviceId = readRegister(0x00);
  Serial.print("ADS1299 Device ID: 0x");
  Serial.println(deviceId, HEX);
  
  if (deviceId != 0x3E) { // Expected device ID for ADS1299
    Serial.println("Warning: Unexpected device ID");
  }
  
  // Set gain and configure channels
  setGain(GAIN_24X); // 24x gain as per MATLAB analysis
  configureChannels();
  
  initialized_ = true;
  Serial.println("ADS1299 Custom driver initialized");
  
  return true;
}

void ADS1299_Custom::startAcquisition() {
  if (!initialized_) return;
  
  // Attach interrupt for data ready
  attachInterrupt(digitalPinToInterrupt(ADS1299_DRDY), dataReadyISR, FALLING);
  
  // Start conversion
  digitalWriteFast(ADS1299_START_PIN, HIGH);
  
  // Enable read data continuous mode
  sendCommand(ADS1299_RDATAC);
  
  Serial.println("ADS1299 data acquisition started");
}

void ADS1299_Custom::stopAcquisition() {
  if (!initialized_) return;
  
  // Stop conversion
  digitalWriteFast(ADS1299_START_PIN, LOW);
  
  // Disable read data continuous mode
  sendCommand(ADS1299_SDATAC);
  
  // Detach interrupt
  detachInterrupt(digitalPinToInterrupt(ADS1299_DRDY));
  
  Serial.println("ADS1299 data acquisition stopped");
}

bool ADS1299_Custom::dataReady() {
  return newDataAvailable_;
}

void ADS1299_Custom::readChannelData(int32_t* channelData) {
  if (!initialized_) return;
  
  SPI.beginTransaction(spiSettings_);
  delayNanoseconds(10);
  digitalWriteFast(ADS1299_CS1, LOW);
  delayNanoseconds(6);
  
  // Skip status registers (3 bytes)
  SPI.transfer(0x00);
  SPI.transfer(0x00);
  SPI.transfer(0x00);
  
  // Read channel data (24-bit per channel)
  for (int i = 0; i < ADS1299_CHANNELS; i++) {
    int32_t sample = 0;
    sample = SPI.transfer(0x00);
    sample = (sample << 8) | SPI.transfer(0x00);
    sample = (sample << 8) | SPI.transfer(0x00);
    
    // Sign extension for 24-bit to 32-bit
    sample = (sample & 0x800000) ? (sample | 0xFF000000) : sample;
    channelData[i] = sample;
  }
  
  delayMicroseconds(2);
  digitalWriteFast(ADS1299_CS1, HIGH);
  SPI.endTransaction();
}

void ADS1299_Custom::readChannelDataVolts(float* channelData) {
  int32_t rawData[ADS1299_CHANNELS];
  readChannelData(rawData);
  
  for (int i = 0; i < ADS1299_CHANNELS; i++) {
    channelData[i] = convertToMicrovolts(rawData[i]);
  }
}

bool ADS1299_Custom::getLatestData(int32_t* channelData) {
  if (!newDataAvailable_) return false;
  
  // Copy data from ISR buffer
  noInterrupts();
  for (int i = 0; i < ADS1299_CHANNELS; i++) {
    channelData[i] = isrBuffer_[i];
  }
  newDataAvailable_ = false;
  interrupts();
  
  return true;
}

void ADS1299_Custom::setGain(uint8_t gain) {
  currentGain_ = gain;
  updateGainMultiplier();
}

void ADS1299_Custom::setSampleRate(uint8_t sampleRate) {
  uint8_t config1 = readRegister(CONFIG1_REG);
  config1 = (config1 & 0xF8) | sampleRate; // Clear and set sample rate bits
  writeRegister(CONFIG1_REG, config1);
}

void ADS1299_Custom::configureChannels() {
  // Configure all channels with current gain and normal input
  uint8_t channelSetting = currentGain_ | CHSET_INPUT;
  
  writeRegister(CH1SET_REG, channelSetting);
  writeRegister(CH2SET_REG, channelSetting);
  writeRegister(CH3SET_REG, channelSetting);
  writeRegister(CH4SET_REG, channelSetting);
  writeRegister(CH5SET_REG, channelSetting);
  writeRegister(CH6SET_REG, channelSetting);
  writeRegister(CH7SET_REG, channelSetting);
  writeRegister(CH8SET_REG, channelSetting);
}

void ADS1299_Custom::writeRegister(uint8_t reg, uint8_t value) {
  SPI.beginTransaction(spiSettings_);
  digitalWriteFast(ADS1299_CS1, LOW);
  delayNanoseconds(6);
  
  SPI.transfer(reg | 0x40); // Write command
  delayMicroseconds(2);
  SPI.transfer(0x00); // Number of bytes - 1
  delayMicroseconds(2);
  SPI.transfer(value);
  
  delayMicroseconds(2);
  digitalWriteFast(ADS1299_CS1, HIGH);
  SPI.endTransaction();
}

uint8_t ADS1299_Custom::readRegister(uint8_t reg) {
  uint8_t value = 0;
  
  SPI.beginTransaction(spiSettings_);
  digitalWriteFast(ADS1299_CS1, LOW);
  delayNanoseconds(6);
  
  SPI.transfer(reg | 0x20); // Read command
  delayMicroseconds(2);
  SPI.transfer(0x00); // Number of bytes - 1
  delayMicroseconds(2);
  value = SPI.transfer(0x00);
  
  delayMicroseconds(2);
  digitalWriteFast(ADS1299_CS1, HIGH);
  SPI.endTransaction();
  
  return value;
}

void ADS1299_Custom::dataReadyISR() {
  // Read data in interrupt service routine
  SPI.beginTransaction(SPISettings(20E6, MSBFIRST, SPI_MODE1));
  delayNanoseconds(10);
  digitalWriteFast(ADS1299_CS1, LOW);
  delayNanoseconds(6);
  
  // Skip status registers
  SPI.transfer(0x00);
  SPI.transfer(0x00);
  SPI.transfer(0x00);
  
  // Read channel data
  for (int i = 0; i < ADS1299_CHANNELS; i++) {
    int32_t sample = 0;
    sample = SPI.transfer(0x00);
    sample = (sample << 8) | SPI.transfer(0x00);
    sample = (sample << 8) | SPI.transfer(0x00);
    
    // Sign extension
    sample = (sample & 0x800000) ? (sample | 0xFF000000) : sample;
    isrBuffer_[i] = sample;
  }
  
  delayMicroseconds(2);
  digitalWriteFast(ADS1299_CS1, HIGH);
  SPI.endTransaction();
  
  newDataAvailable_ = true;
}

void ADS1299_Custom::reset() {
  sendCommand(ADS1299_RESET);
  delayMicroseconds(9); // Wait for reset
}

void ADS1299_Custom::powerUp() {
  digitalWriteFast(ADS1299_PWDN, HIGH);
  delay(128); // Wait for power-on reset
}

void ADS1299_Custom::sendCommand(uint8_t command) {
  SPI.beginTransaction(spiSettings_);
  digitalWriteFast(ADS1299_CS1, LOW);
  delayNanoseconds(6);
  SPI.transfer(command);
  delayMicroseconds(2);
  digitalWriteFast(ADS1299_CS1, HIGH);
  SPI.endTransaction();
}

void ADS1299_Custom::updateGainMultiplier() {
  // Calculate gain multiplier based on current gain setting
  // From ADS1299 datasheet and MATLAB conversion formula
  const float vref = 4.5f; // Reference voltage
  const float fullScale = (1UL << 24); // 24-bit full scale
  
  int gainValue = 1;
  switch (currentGain_) {
    case GAIN_1X:  gainValue = 1;  break;
    case GAIN_2X:  gainValue = 2;  break;
    case GAIN_4X:  gainValue = 4;  break;
    case GAIN_6X:  gainValue = 6;  break;
    case GAIN_8X:  gainValue = 8;  break;
    case GAIN_12X: gainValue = 12; break;
    case GAIN_24X: gainValue = 24; break;
    default: gainValue = 24; break;
  }
  
  // MATLAB formula: x * ((2.*4.5)./gain)./(2.^24).*1e6
  gainMultiplier_ = (2.0f * vref / gainValue) / fullScale * 1000000.0f; // µV
}

float ADS1299_Custom::convertToMicrovolts(int32_t adcValue) {
  return (float)adcValue * gainMultiplier_;
}