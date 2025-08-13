#include <Arduino.h>

void setup() {
  Serial.begin(115200);
  pinMode(13, OUTPUT); // Built-in LED
  
  // Wait a bit for serial to initialize
  delay(2000);
  
  Serial.println("=== SIMPLE TEENSY TEST ===");
  Serial.println("If you see this, basic communication works!");
  Serial.println("LED should be blinking...");
}

void loop() {
  static unsigned long lastTime = 0;
  static int counter = 0;
  
  // Blink LED and print every second
  if (millis() - lastTime > 1000) {
    lastTime = millis();
    counter++;
    
    // Toggle LED
    digitalWrite(13, !digitalRead(13));
    
    // Print status
    Serial.print("Heartbeat #");
    Serial.print(counter);
    Serial.print(" - Time: ");
    Serial.print(millis());
    Serial.println(" ms");
  }
  
  // Echo any received characters
  if (Serial.available()) {
    char c = Serial.read();
    Serial.print("Received: '");
    Serial.print(c);
    Serial.println("'");
  }
}