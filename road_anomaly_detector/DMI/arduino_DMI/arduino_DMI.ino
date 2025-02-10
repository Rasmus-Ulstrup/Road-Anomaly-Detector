// Pin definitions for the encoder channels and output
const int encoderPinA = 2;  
const int encoderPinB = 3;  
const int lineRatePinA = 8;  
const int lineRatePinB = 9;
// Constants
const int Resolution_H = 2048;    // 
const float Pixel_size = 7;     // P
const float Focal = 8.5;          // 
const float defaultWD = 0.915;      // D

// Function prototype
float calculateSpatialResolution(float WD);
void updateEncoder();//
float WD;                         // 
float pixelSize_mm;               //
float sensorWidth_mm;             //
float fieldOfView;                // 
float spatialResolution;          // 

volatile int16_t encoderTicks = 0;
static int16_t encoderThreshold = 4*2;
int16_t encoderAll = 0;


unsigned long lastTriggerTime_light = 0;
unsigned long lastTriggerTime_camera = 0;
const unsigned long pulseDurationMicros = 120;  //

void setup() {
  DDRB |= (1<<PB0) | (1<<PB1);
  Serial.begin(9600);
  Serial.println("Enter the working distance (WD) in meters (or wait 30 seconds for default):");

  unsigned long startTime = millis();
  float WD = 0.0;
  bool inputReceived = false;

  // Wait for input or timeout after 30 seco
  while (millis() - startTime < 100) {
    if (Serial.available() > 0) {
      WD = Serial.parseFloat();
      inputReceived = true;
      break;
    }
  }
  // Use default WD if no input is received
  if (!inputReceived) {
    WD = defaultWD;
    Serial.println("No input received. Using default Working Distance (WD): ");
  } else {
    Serial.println("Working Distance (WD) entered: ");
  }
  Serial.println(WD);

  // Calculate the spatial resolution using the function
  float spatialResolution = calculateSpatialResolution(WD);
  Serial.print("Spatial Resolution (m/pixel): ");
  Serial.println(spatialResolution, 12);  // 6 decimal places for precision

  // pinMode(lineRatePinA, OUTPUT);
  // pinMode(lineRatePinB, OUTPUT);
  pinMode(encoderPinA, INPUT_PULLUP);
  pinMode(encoderPinB, INPUT_PULLUP);

  // Attach interrupts for both encoder pins to maximize resolution
  attachInterrupt(digitalPinToInterrupt(encoderPinA), updateEncoder, RISING);
  attachInterrupt(digitalPinToInterrupt(encoderPinB), updateEncoder, RISING);
}

void loop() {
  if (encoderTicks >= encoderThreshold) {
    encoderTicks = 0;  // Reset encoder count
    digitalWrite(53,1);
    digitalWrite(52,1);
    // PORTB |= (1<<PB0) | (1<<PB1);
    // PORTB &= ~(1<<PB1);
    digitalWrite(52,0);
    lastTriggerTime_light = micros();  // Record the time of the trigger
  }

  // Check if it's time to pull the lineRatePin back to LOW
  if  ((PORTB & (1 << PB0)) && (micros() - lastTriggerTime_light) >= pulseDurationMicros) {
    // PORTB &=~(1<<PB0); //53 - lys
    // PORTB |= (1<<PB1); //52 - cam
    // PORTB &=~(1<<PB1);
    digitalWrite(53,0);
    digitalWrite(52,1);
    digitalWrite(52,0);
    lastTriggerTime_light=0;
  }
}
void updateEncoder() {
  encoderTicks++;
}

float calculateSpatialResolution(float WD) {
  float pixelSize_mm = Pixel_size / 1000.0;
  float sensorWidth_mm = Resolution_H * pixelSize_mm;
  float fieldOfView = (WD * sensorWidth_mm) / Focal;
  return fieldOfView / Resolution_H;
}
