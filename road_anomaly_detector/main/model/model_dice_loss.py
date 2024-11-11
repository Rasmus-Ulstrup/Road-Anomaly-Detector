import tensorflow as tf
import keras
import keras_cv
import os
print(tf.__version__)
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
def preprocess_crack_segmentation(inputs):
    def unpackage(image, mask):
        return {
            "images": image,
            "segmentation_masks": mask,
        }

    # Map over the dataset and apply unpackage function
    outputs = inputs.map(lambda image, mask: unpackage(image, mask))
    
    # Apply resizing and batching
    outputs = outputs.map(keras_cv.layers.Resizing(height=512, width=512))
    outputs = outputs.batch(4, drop_remainder=True)
    
    return outputs

train_ds = preprocess_crack_segmentation(train_ds)
eval_ds = preprocess_crack_segmentation(eval_ds)

# Step 4: Data Augmentation
def augment_data(dataset):
    return dataset.map(keras_cv.layers.RandomFlip())

# Step 5: Dice Coefficient and Dice Loss
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2. * intersection + smooth) / (union + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

# Step 6: Model Configuration
BATCH_SIZE = 4
INITIAL_LR = 0.007 * BATCH_SIZE / 16
EPOCHS = 10  # Adjust as needed
NUM_CLASSES = 2  # Since you have cracks and background (2 classes)
learning_rate = keras.optimizers.schedules.CosineDecay(
    INITIAL_LR,
    decay_steps=EPOCHS * 2124,
)

model = keras_cv.models.DeepLabV3Plus.from_preset(
    "resnet50_v2_imagenet", num_classes=NUM_CLASSES
)

# Step 7: Compile the Model with Dice Loss
model.compile(
    optimizer=keras.optimizers.SGD(
        learning_rate=learning_rate, weight_decay=0.0001, momentum=0.9, clipnorm=10.0
    ),
    loss=dice_loss,  # Use Dice loss instead of Categorical Crossentropy
    metrics=[
        keras.metrics.MeanIoU(num_classes=NUM_CLASSES, sparse_y_true=False, sparse_y_pred=False),
        keras.metrics.CategoricalAccuracy(),
    ],
)

# Step 8: Training the Model
def dict_to_tuple(x):
    return x["images"], tf.one_hot(
        tf.cast(tf.squeeze(x["segmentation_masks"], axis=-1), "int32"), NUM_CLASSES
    )

train_ds = augment_data(train_ds)
train_ds = train_ds.map(dict_to_tuple)
eval_ds = eval_ds.map(dict_to_tuple)

# Train the model on GPU (if available)
model.fit(train_ds, validation_data=eval_ds, epochs=EPOCHS)

# Step 9: Testing the Model
test_ds = load_crack_segmentation_dataset("C:/Users/bjs/Downloads/crack_segmentation_dataset/crack_segmentation_dataset/test/images",
                                          "C:/Users/bjs/Downloads/crack_segmentation_dataset/crack_segmentation_dataset/test/masks")
test_ds = preprocess_crack_segmentation(test_ds)

images, masks = next(iter(test_ds.take(1)))
preds = model(images)
preds = tf.argmax(preds, axis=-1, output_type=tf.int32)

# Visualize predictions
keras_cv.visualization.plot_segmentation_mask_gallery(
    images,
    value_range=(0, 255),
    num_classes=NUM_CLASSES,
    y_true=masks,
    y_pred=preds,
    scale=3,
    rows=1,
    cols=4,
)
