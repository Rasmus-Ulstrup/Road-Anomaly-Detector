import tensorflow as tf
import keras_cv
import keras
import os
# Step 2: Dataset Loading and Preprocessing
def load_crack_segmentation_dataset(image_dir, mask_dir):
    images = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
    masks = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)])
    
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    
    def load_image(image_path, mask_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)  # assuming images are in JPEG format
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)  # assuming masks are in PNG format
        return image, mask
    
    return dataset.map(load_image)

# Load training and evaluation datasets
train_images_dir = r"C:/Users/bjs/Downloads/crack_segmentation_dataset/crack_segmentation_dataset/train/images"
train_masks_dir = r"C:/Users/bjs/Downloads/crack_segmentation_dataset/crack_segmentation_dataset/train/masks"
eval_images_dir = r"C:/Users/bjs/Downloads/crack_segmentation_dataset/crack_segmentation_dataset/test/images"
eval_masks_dir = "C:/Users/bjs/Downloads/crack_segmentation_dataset/crack_segmentation_dataset/test/masks"

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

# Step 4 Data Augmentation
train_ds = train_ds.map(keras_cv.layers.RandomFlip())

# Step 5 Model Configuration
BATCH_SIZE = 4
INITIAL_LR = 0.007 * BATCH_SIZE / 16
EPOCHS = 10  # You can adjust the number of epochs as needed
NUM_CLASSES = 2  # Since you have cracks and background (2 classes)
learning_rate = keras.optimizers.schedules.CosineDecay(
    INITIAL_LR,
    decay_steps=EPOCHS * 2124,
)

model = keras_cv.models.DeepLabV3Plus.from_preset(
    "resnet50_v2_imagenet", num_classes=NUM_CLASSES
)

#Step 6: Compile the Model
model.compile(
    optimizer=keras.optimizers.SGD(
        learning_rate=learning_rate, weight_decay=0.0001, momentum=0.9, clipnorm=10.0
    ),
    loss=keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=[
        keras.metrics.MeanIoU(
            num_classes=NUM_CLASSES, sparse_y_true=False, sparse_y_pred=False
        ),
        keras.metrics.CategoricalAccuracy(),
    ],
)

#Step 7: Training the Model
def dict_to_tuple(x):
    import tensorflow as tf
    return x["images"], tf.one_hot(
        tf.cast(tf.squeeze(x["segmentation_masks"], axis=-1), "int32"), NUM_CLASSES
    )

train_ds = train_ds.map(dict_to_tuple)
eval_ds = eval_ds.map(dict_to_tuple)

model.fit(train_ds, validation_data=eval_ds, epochs=EPOCHS)

#Step 8 Testing the Model
test_ds = load_crack_segmentation_dataset("path/to/test/images", "path/to/test/masks")
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