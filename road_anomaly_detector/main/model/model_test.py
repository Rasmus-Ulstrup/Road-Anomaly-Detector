from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import Adam

# Custom Dice Loss and Dice Coefficient (with smoothing to avoid NaN)
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

# Load the trained model
model = load_model("C:/Thesis/Road-Anomaly-Detector/crack_detection_model.h5", 
                   custom_objects={"dice_loss": dice_loss, "dice_coefficient": dice_coefficient})

# Check for NaN in model weights
for layer in model.layers:
    if tf.reduce_any(tf.math.is_nan(layer.get_weights())):
        print(f"NaN found in layer: {layer.name}")

# Set a lower learning rate and use gradient clipping
optimizer = Adam(learning_rate=1e-5, clipvalue=1.0)  # Use smaller learning rate and gradient clipping
model.compile(optimizer=optimizer, loss=dice_loss, metrics=[dice_coefficient])

# Preprocess image function
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # assuming JPEG format
    image = tf.image.resize(image, [448, 448])
    image = tf.cast(image, tf.float32) / 255.0  # normalize to [0,1]
    image = tf.expand_dims(image, axis=0)  # add batch dimension
    return image

image_path = r"C:\Thesis\Road-Anomaly-Detector\road_anomaly_detector\main\model\forest_003.jpg"
preprocessed_image = preprocess_image(image_path)
prediction = model.predict(preprocessed_image)

# Threshold the prediction to create a binary mask (if needed)
binary_mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)

# Display the original image and the predicted mask
original_image = tf.image.decode_jpeg(tf.io.read_file(image_path))
original_image = tf.image.resize(original_image, [448, 448])

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_image.numpy().astype("uint8"))

plt.subplot(1, 2, 2)
plt.title("Predicted Crack Mask")
plt.imshow(binary_mask, cmap="gray")

plt.show()
