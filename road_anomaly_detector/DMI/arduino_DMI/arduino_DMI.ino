// Pin definitions for the encoder channels and output
const int encoderPinA = 2;  
const int encoderPinB = 3;  
const int lineRatePin = 8;  

// Constants
constexpr bool debugEnable = true;
constexpr int TICKS_PER_REVOLUTION = 2000;

void setup() {
  Serial.begin(9600);

  pinMode(lineRatePin, OUTPUT);
  pinMode(encoderPinA, INPUT_PULLUP);
  pinMode(encoderPinB, INPUT_PULLUP);

  // Attach interrupts for both encoder pins to maximize resolution
  attachInterrupt(digitalPinToInterrupt(encoderPinA), updateEncoder, CHANGE);
  attachInterrupt(digitalPinToInterrupt(encoderPinB), updateEncoder, CHANGE);
}

void loop() {
  unsigned long currentTime = micros();
  unsigned long timeElapsed = currentTime - lastTime;

}

void updateEncoder() {
  encoderTicks++
}
