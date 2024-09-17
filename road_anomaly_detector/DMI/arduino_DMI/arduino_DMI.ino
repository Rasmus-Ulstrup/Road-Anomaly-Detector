// Pin definitions for the encoder channels
const int encoderPinA = 2;  // Interrupt pin 0 on Arduino (pin 2)
const int encoderPinB = 3;  // Interrupt pin 1 on Arduino (pin 3)
const int lineRatePin = 8;  // 8

// Define constants
#define debugEnable 1 //1 for enable 0 for disable
#define WHEEL_DIAMETER 0.20         // meter
#define TICKS_PER_REVOLUTION 2000  // ticks per revolution
#define PI 3.14159265359

// Camera constants:
#define SPATIAL_RESOLUTION_MM 0.3541176470588235 //mm/pixel

// Calculate derived constants
#define WHEEL_CIRCUMFERENCE (PI * WHEEL_DIAMETER)      // meter
#define DISTANCE_PER_TICK (WHEEL_CIRCUMFERENCE / TICKS_PER_REVOLUTION)  // meters per tick

// Variables to store encoder state
volatile long encoderTicks = 0;
volatile int lastEncoded = 0;
unsigned long halfPeriod;          // Half of the square wave period in microseconds

// Variables for speed and distance
unsigned long lastTime = 0;
float speed = 0;     // meters per second
float totalDistance = 0; // meters

const float wheel_circ = PI * WHEEL_DIAMETER; //meter
const float distPerTick = wheel_circ / TICKS_PER_REVOLUTION;

float lineRate = 0; // hz
const float spatialRes = SPATIAL_RESOLUTION_MM / 1000; //m/pixel

// Variables for square wave generation
unsigned long previousMicros = 0;  // Store the last time the pin was toggled

void setup() {
  // Initialize serial communication for debugging
  Serial.begin(9600);
  
  // Set up lineRate pin
  pinMode(lineRatePin, OUTPUT);  // sets the pin as output

  // Set up encoder pins as input
  pinMode(encoderPinA, INPUT);
  pinMode(encoderPinB, INPUT);

  // Enable pullup resistors on encoder pins if needed
  digitalWrite(encoderPinA, HIGH);
  digitalWrite(encoderPinB, HIGH);

  // Attach interrupts to encoder pins
  attachInterrupt(digitalPinToInterrupt(encoderPinA), updateEncoder, CHANGE);
  attachInterrupt(digitalPinToInterrupt(encoderPinB), updateEncoder, CHANGE);
}

void loop() {
  // Calculate speed
  unsigned long currentTime = micros();
  unsigned long timeElapsed = currentTime - lastTime;

  if (timeElapsed > 250000) {  // Calculate every 250ms
    // Speed in meters per second
    speed = (encoderTicks * distPerTick) / (timeElapsed / 1000000.0);

    // Add distance covered to total distance
    totalDistance += encoderTicks * distPerTick;

    // Calculate line rate
    lineRate = abs(speed / spatialRes);
    if (lineRate >= 1) {
      halfPeriod = (1000000 / lineRate) / 2;  // Half period in microseconds
    }

    if (debugEnable == 1){
      // Reset for the next loop
      Serial.print("Encoder Ticks: ");
      Serial.print(encoderTicks);
      Serial.print("\t");
      encoderTicks = 0;
      lastTime = currentTime;

      // Print the speed and total distance
      Serial.print("Speed: ");
      Serial.print(speed,6);
      Serial.print(" m/s\t\t");

      Serial.print("Total Distance: ");
      Serial.print(totalDistance);
      Serial.print(" meters\t");

      Serial.print(lineRate,2);
      Serial.println(" hz");
    }
  }

  // Generate the square wave based on the line rate if it is greater than or equal to 1 Hz
  if (lineRate >= 1) {
    unsigned long currentMicros = micros();
    
    if (currentMicros - previousMicros >= halfPeriod) {
      // Toggle the output pin
      digitalWrite(lineRatePin, !digitalRead(lineRatePin));  // Flip the state of the pin
      
      // Update the previous time
      previousMicros = currentMicros;
    }
  } else {
    // Do nothing if lineRate is below 1 Hz
    digitalWrite(lineRatePin, LOW);  // Set the pin to LOW or keep it in a default state
  }
}

// Function to update the encoder ticks based on changes
void updateEncoder() {
  int MSB = digitalRead(encoderPinA); // Most Significant Bit
  int LSB = digitalRead(encoderPinB); // Least Significant Bit

  int encoded = (MSB << 1) | LSB; // Converting the 2 pin values to a single number
  int sum = (lastEncoded << 2) | encoded; // Tracking the last encoded value
  
  // Increment or decrement the ticks based on the direction
  if (sum == 0b1101 || sum == 0b0100 || sum == 0b0010 || sum == 0b1011) encoderTicks++;
  if (sum == 0b1110 || sum == 0b0111 || sum == 0b0001 || sum == 0b1000) encoderTicks--;
  
  lastEncoded = encoded; // Store this value for the next loop
}
