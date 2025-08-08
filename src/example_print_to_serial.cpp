#include <Arduino.h>
#include <SdFat.h>          // Library for SD card access
#include <RingBuf.h>     // Library for ring buffer

/*
// put function declarations here:
int myFunction(int, int);

void setup() {
  // put your setup code here, to run once:
  int result = myFunction(2, 3);
}

void loop() {
  // put your main code here, to run repeatedly:
}

// put function definitions here:
int myFunction(int x, int y) {
  return x + y;
}
*/

#include <Arduino.h>

void example_setup() {
  Serial.begin(115200);
  // Print log
  Serial.println("setup");
}

float i=0;
void example_loop() {
  i+=0.1;

  // Print log
  Serial.print("loop");
  Serial.println(i);
  
  // Plot a sinus
  Serial.print(">sin:");
  Serial.println(sin(i));

  // Plot a cosinus
  Serial.print(">cos:");
  Serial.println(cos(i));
    
  delay(50);
}