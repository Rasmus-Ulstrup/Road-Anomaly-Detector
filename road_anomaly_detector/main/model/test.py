import tensorflow as tf
import keras
import keras_cv
from tensorflow.keras import layers, models, mixed_precision
import os
from tensorflow.keras.backend import clear_session
clear_session()
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Step 1: GPU Configuration
if tf.config.list_physical_devices('GPU'):
    print("GPU detected!")
else:
    print("No GPU detected, using CPU.")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only allocate memory as needed
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# Step 2: Dataset Loading and Preprocessing
def load_crack_segmentation_dataset(image_dir, mask_dir):
    images = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
    masks = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)])
    
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    
    def load_image(image_path, mask_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)  # Assuming images are in JPEG format
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_jpeg(mask, channels=1)  # Assuming masks are in JPEG format
        return image, mask
    
    return dataset.map(load_image)

# Directories for images and masks (update with actual paths)
train_images_dir = r"C:/Users/bjs/Downloads/crack_segmentation_dataset/crack_segmentation_dataset/train/images"
train_masks_dir = r"C:/Users/bjs/Downloads/crack_segmentation_dataset/crack_segmentation_dataset/train/masks"
eval_images_dir = r"C:/Users/bjs/Downloads/crack_segmentation_dataset/crack_segmentation_dataset/test/images"
eval_masks_dir = r"C:/Users/bjs/Downloads/crack_segmentation_dataset/crack_segmentation_dataset/test/masks"

train_ds = load_crack_segmentation_dataset(train_images_dir, train_masks_dir)
eval_ds = load_crack_segmentation_dataset(eval_images_dir, eval_masks_dir)

# Step 3: Preprocessing the Data
def preprocess_crack_segmentation(dataset):
    def resize_function(image, mask):
        # Resize both image and mask
        image_resized = keras_cv.layers.Resizing(height=512, width=512)(image)
        mask_resized = keras_cv.layers.Resizing(height=512, width=512)(mask)
        return image_resized, mask_resized

    dataset = dataset.map(resize_function)
    return dataset

train_ds = preprocess_crack_segmentation(train_ds)
eval_ds = preprocess_crack_segmentation(eval_ds)

# Step 4: Data Augmentation
tf.config.run_functions_eagerly(True)

def augment_fn(image, mask):
    # Set 'training' to True for augmentation
    image = keras_cv.layers.RandomRotation(0.2)(image, training=True)
    mask = keras_cv.layers.RandomRotation(0.2)(mask, training=True)
    return image, mask

def augment_data(dataset):
    return dataset.map(augment_fn)
#yeyeye
# Apply augmentation on your dataset
train_ds = augment_data(train_ds)

# Step 5: Dice Coefficient and Dice Loss
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2. * intersection + smooth) / (union + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

# Step 6: Model Configuration
BATCH_SIZE = 2
INITIAL_LR = 0.007 * BATCH_SIZE / 16
EPOCHS = 10  # Adjust as needed
NUM_CLASSES = 1  # Binary segmentation (crack vs background)
learning_rate = tf.keras.optimizers.schedules.CosineDecay(
    INITIAL_LR,
    decay_steps=EPOCHS * 2124,
)

def unet_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    # Contracting path (Encoder)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    
    # Bottleneck
    b = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    b = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(b)
    
    # Expansive path (Decoder)
    u4 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(b)
    u4 = layers.concatenate([u4, c4])
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u4)
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c5)
    
    u3 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u3 = layers.concatenate([u3, c3])
    c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u3)
    c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c6)
    
    u2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u2 = layers.concatenate([u2, c2])
    c7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u2)
    c7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c7)
    
    u1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u1 = layers.concatenate([u1, c1])
    c8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u1)
    c8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c8)
    
    # Output layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(c8)  # For binary segmentation
    
    model = models.Model(inputs, outputs)
    return model

# Create U-Net model
input_shape = (512, 512, 3)  # Change input shape based on your dataset
num_classes = 1  # Set to 1 for binary segmentation
model = unet_model(input_shape, num_classes)

# Step 7: Compile the Model with Dice Loss
model.compile(optimizer='adam', loss=dice_loss, metrics=[dice_coefficient])

# Display model summary
model.summary()

# Step 8: Training the Model
def dict_to_tuple(x):
    return x["images"], tf.one_hot(
        tf.cast(x["class_ids"], tf.int32), NUM_CLASSES
    )

train_dataset = (
    train_ds.shuffle(256)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(tf.data.experimental.AUTOTUNE)
)
val_dataset = (
    eval_ds.shuffle(256)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(tf.data.experimental.AUTOTUNE)
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True)],
)

# Optional: Save the trained model
model.save("crack_detection_model.h5")
