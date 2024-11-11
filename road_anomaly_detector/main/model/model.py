import keras
from keras import ops

import keras_cv
import numpy as np

from keras_cv.datasets.pascal_voc.segmentation import load as load_voc

"""
## Perform semantic segmentation with a pretrained DeepLabv3+ model

The highest level API in the KerasCV semantic segmentation API is the `keras_cv.models`
API. This API includes fully pretrained semantic segmentation models, such as
`keras_cv.models.DeepLabV3Plus`.

Let's get started by constructing a DeepLabv3+ pretrained on the pascalvoc dataset.
"""

model = keras_cv.models.DeepLabV3Plus.from_preset(
    "deeplab_v3_plus_resnet50_pascalvoc",
    num_classes=21,
    input_shape=[512, 512, 3],
)

"""
Let us visualize the results of this pretrained model
"""

filepath = r"C:\Thesis\Road-Anomaly-Detector\road_anomaly_detector\main\model\gCNcJJI.jpg"
image = keras.utils.load_img(filepath)

resize = keras_cv.layers.Resizing(height=512, width=512)
image = resize(image)
image = keras.ops.expand_dims(np.array(image), axis=0)
preds = ops.expand_dims(ops.argmax(model(image), axis=-1), axis=-1)
keras_cv.visualization.plot_segmentation_mask_gallery(
    image,
    value_range=(0, 255),
    num_classes=1,
    y_true=None,
    y_pred=preds,
    scale=3,
    rows=1,
    cols=1,
)

