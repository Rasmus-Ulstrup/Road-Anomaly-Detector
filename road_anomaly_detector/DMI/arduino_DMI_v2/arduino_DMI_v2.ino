// Pin definitions for the encoder channels and output
const int encoderPinA = 2;  
const int encoderPinB = 3;  
const int lineRatePin = 8;  

// Constants
constexpr bool debugEnable = true;
constexpr int TICKS_PER_REVOLUTION = 2000;

const int Resolution_H = 4096;    // Horizontal resolution in pixels
const float Pixel_size = 3.5;     // Pixel size in micrometers (µm)
const float Focal = 8.5;          // Focal length in millimeters (mm)
const float defaultWD = 0.8;      // Default working distance in meters

// Function prototype
float calculateSpatialResolution(float WD);

// Variables to store calculated values
float WD;                         // Working distance in meters, input from user
float pixelSize_mm;               // Pixel size in millimeters
float sensorWidth_mm;             // Sensor width in millimeters
float fieldOfView;                // Field of view in millimeters
float spatialResolution;          // Spatial resolution in meters per pixel

volatile float encoderTicks = 0;
static float encoderThreshold = 200;


unsigned long lastTriggerTime = 0;
const unsigned long pulseDurationMicros = 10;  // 10 microseconds for the pulse
void setup() {
  Serial.begin(9600);
  Serial.println("Enter the working distance (WD) in meters (or wait 30 seconds for default):");

  unsigned long startTime = millis();
  float WD = 0.0;
  bool inputReceived = false;

  // Wait for input or timeout after 30 seconds (30000 milliseconds)
  while (millis() - startTime < 30000) {
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

  pinMode(lineRatePin, OUTPUT);
  pinMode(encoderPinA, INPUT_PULLUP);
  pinMode(encoderPinB, INPUT_PULLUP);

  // Attach interrupts for both encoder pins to maximize resolution
  attachInterrupt(digitalPinToInterrupt(encoderPinA), updateEncoder, CHANGE);
  attachInterrupt(digitalPinToInterrupt(encoderPinB), updateEncoder, CHANGE);
}

void loop() {
  if (encoderTicks >= encoderThreshold) {
    encoderTicks = 0;  // Reset encoder count
    digitalWrite(lineRatePin, HIGH);  // Trigger camera
    lastTriggerTime = micros();  // Record the time of the trigger
  }

  // Check if it's time to pull the lineRatePin back to LOW
  if (digitalRead(lineRatePin) == HIGH && (micros() - lastTriggerTime) >= pulseDurationMicros) {
    digitalWrite(lineRatePin, LOW);  // Reset trigger
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
