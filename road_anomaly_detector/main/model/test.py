import tensorflow as tf
import keras
from tensorflow.keras import layers, models, mixed_precision
import os
from tensorflow.keras.backend import clear_session
clear_session()

# Set mixed precision policy for better GPU utilization
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Step 1: GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU detected! Enabling memory growth.")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU detected. Using CPU.")

# Step 2: Dataset Loading and Preprocessing
all_images_dir = r"C:\Users\bjs\Downloads\datasets\forest\Images"
all_masks_dir = r"C:\Users\bjs\Downloads\datasets\forest\Masks"

def load_and_split_dataset(image_dir, mask_dir, train_split=0.8):
    images = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
    masks = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)])
    
    dataset_size = len(images)
    split_index = int(train_split * dataset_size)
    
    # Split into training and evaluation datasets
    train_images, eval_images = images[:split_index], images[split_index:]
    train_masks, eval_masks = masks[:split_index], masks[split_index:]
    
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
    eval_ds = tf.data.Dataset.from_tensor_slices((eval_images, eval_masks))
    
    def load_image(image_path, mask_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32) / 255.0  # Normalize image
        
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_jpeg(mask, channels=1)
        mask = tf.cast(mask, tf.float32) / 255.0  # Normalize mask
        
        return image, mask
    
    return train_ds.map(load_image), eval_ds.map(load_image)

# Load and split the dataset
train_ds, eval_ds = load_and_split_dataset(all_images_dir, all_masks_dir)

# Step 3: Preprocessing the Data
def preprocess_crack_segmentation(dataset):
    def resize_function(image, mask):
        image_resized = tf.image.resize(image, [448, 448])
        mask_resized = tf.image.resize(mask, [448, 448])
        return image_resized, mask_resized

    dataset = dataset.map(resize_function)
    return dataset

train_ds = preprocess_crack_segmentation(train_ds)
eval_ds = preprocess_crack_segmentation(eval_ds)

# Step 4: Data Augmentation
def augment_fn(image, mask):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    mask = tf.image.random_flip_left_right(mask)
    mask = tf.image.random_flip_up_down(mask)
    return image, mask

train_ds = train_ds.map(augment_fn)

# Step 5: Dice Coefficient and Dice Loss
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    # Prevent division by zero by ensuring union is never zero
    return (2. * intersection + smooth) / (union + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

# Step 6: Model Configuration
BATCH_SIZE = 2
EPOCHS = 10
NUM_CLASSES = 1

def unet_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    # Contracting path
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
    
    # Expansive path
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
    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(c8)
    model = models.Model(inputs, outputs)
    return model

# Create and compile U-Net model
input_shape = (448, 448, 3)
model = unet_model(input_shape, NUM_CLASSES)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=dice_loss, metrics=[dice_coefficient])

# Step 8: Training the Model
train_dataset = train_ds.shuffle(256).batch(BATCH_SIZE).cache().prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = eval_ds.batch(BATCH_SIZE).cache().prefetch(tf.data.experimental.AUTOTUNE)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True)],
)

# Optional: Save the trained model
model.save("crack_detection_model.h5")