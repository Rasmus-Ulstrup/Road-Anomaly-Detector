// Pin definitions for the encoder channels and output
const int encoderPinA = 2;  
const int encoderPinB = 3;  
const int lineRatePinA = 8;  
const int lineRatePinB = 9;
// Constants
const int Resolution_H = 2048;    // Horizontal resolution in pixels
const float Pixel_size = 7;     // Pixel size in micrometers (µm)
const float Focal = 8.5;          // Focal length in millimeters (mm)
const float defaultWD = 0.915;      // Default working distance in meters

// Function prototype
float calculateSpatialResolution(float WD);

// Variables to store calculated values
float WD;                         // Working distance in meters, input from user
float pixelSize_mm;               // Pixel size in millimeters
float sensorWidth_mm;             // Sensor width in millimeters
float fieldOfView;                // Field of view in millimeters
float spatialResolution;          // Spatial resolution in meters per pixel

volatile int16_t encoderTicks = 0;
volatile int16_t encoderTicks_camera = 0;
static int16_t encoderThreshold = 4*2;
int16_t encoderAll = 0;


unsigned long lastTriggerTime_light = 0;
unsigned long lastTriggerTime_camera = 0;
const unsigned long pulseDurationMicros = 25;  // 10 microseconds for the pulse

void setup() {
  Serial.begin(9600);
  Serial.println("Enter the working distance (WD) in meters (or wait 30 seconds for default):");

  unsigned long startTime = millis();
  float WD = 0.0;
  bool inputReceived = false;

  // Wait for input or timeout after 30 seconds (30000 milliseconds)
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
  pinMode(lineRatePinB, OUTPUT);
  pinMode(encoderPinA, INPUT_PULLUP);
  pinMode(encoderPinB, INPUT_PULLUP);

  // Attach interrupts for both encoder pins to maximize resolution
  attachInterrupt(digitalPinToInterrupt(encoderPinA), updateEncoder, RISING);
  attachInterrupt(digitalPinToInterrupt(encoderPinB), updateEncoder, RISING);
}

void loop() {
  if (encoderTicks >= encoderThreshold) {
    encoderTicks = 0;  // Reset encoder count
    digitalWrite(lineRatePinA, HIGH);  // Trigger camera
    lastTriggerTime_light = micros();  // Record the time of the trigger
  }
  if (encoderTicks_camera>=encoderThreshold) {
    encoderTicks_camera=0;
    digitalWrite(lineRatePinB, HIGH);
    lastTriggerTime_camera = micros();
  }

  // Check if it's time to pull the lineRatePin back to LOW
  if (digitalRead(lineRatePinA) == HIGH && (micros() - lastTriggerTime_light) >= pulseDurationMicros) {
    digitalWrite(lineRatePinA, LOW);  // Reset trigger
    lastTriggerTime_light=0;
  }
  if (digitalRead(lineRatePinB) == HIGH && (micros() - lastTriggerTime_camera) >= pulseDurationMicros) {
    digitalWrite(lineRatePinB, LOW);  // Reset trigger
    lastTriggerTime_camera=0;
  }
}

void updateEncoder() {
  encoderTicks++;
  encoderTicks_camera+=2;
}

float calculateSpatialResolution(float WD) {
  float pixelSize_mm = Pixel_size / 1000.0;
  float sensorWidth_mm = Resolution_H * pixelSize_mm;
  float fieldOfView = (WD * sensorWidth_mm) / Focal;
  return fieldOfView / Resolution_H;
}
