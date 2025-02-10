// Pin definitions for the encoder channels and output
const int encoderPinA = 2;  
const int encoderPinB = 3;  
const int lineRatePinA = 8;  
const int lineRatePinB = 9;
// Constants
const int Resolution_H = 4096;    
const float Pixel_size = 3.5;     
const float Focal = 8.5;          // 
const float defaultWD = 0.784;      // 

// Function prototype
float calculateSpatialResolution(float WD);

// Variables to store calculated values
float WD;                         //
float pixelSize_mm;               // 
float sensorWidth_mm;             // S
float fieldOfView;                // 
float spatialResolution;          // 

volatile int16_t encoderTicks = 0;
static int16_t encoderThreshold = 4*2;
int16_t encoderAll = 0;


unsigned long lastTriggerTime = 0;
const unsigned long pulseDurationMicros = 150;  // 10 microseconds for the pulse

void setup() {
  Serial.begin(9600);
  Serial.println("Enter the working distance (WD) in meters (or wait 30 seconds for default):");

  unsigned long startTime = millis();
  float WD = 0.0;
  bool inputReceived = false;

  // Wait for input or timeout after 30
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

  pinMode(lineRatePinA, OUTPUT);
  pinMode(encoderPinA, INPUT_PULLUP);
  pinMode(encoderPinB, INPUT_PULLUP);

  // Attach interrupts for both encoder pin
  attachInterrupt(digitalPinToInterrupt(encoderPinA), updateEncoder, RISING);
  attachInterrupt(digitalPinToInterrupt(encoderPinB), updateEncoder, RISING);
}

void loop() {
  if (encoderTicks >= encoderThreshold) {
    encoderTicks = 0;  // Reset encoder count
    digitalWrite(lineRatePinA, HIGH);  // Trigger camera
    lastTriggerTime = micros();  // Record the time of the trigger
  }

  //
  if (digitalRead(lineRatePinA) == HIGH && (micros() - lastTriggerTime) >= pulseDurationMicros) {
    digitalWrite(lineRatePinA, LOW);  // Reset trigger
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
