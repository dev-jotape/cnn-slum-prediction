import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow.keras.utils as utils
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import rasterio
import numpy as np
import os
from PIL import Image 

print('IMPORT DATA -----------------------------')

image_format = 'tiff'

version = 'v1'
source_city = 'sp'
target_city = 'rj'
source = 'sentinel2' # 'gmaps'
target_dataset = '{}_2024'.format(target_city.upper())
target_data_dir = '../../dataset/{}/{}'.format(source, target_dataset)

input_shape = (224, 224, 3)

def load_tiff_image(file_path):
    with rasterio.open(file_path) as src:
        image = src.read([1, 2, 3])  # Read the first three bands
        image = np.moveaxis(image, 0, -1)  # Move bands to the last dimension
        upscaled_image = Image.fromarray(np.uint8(image*255)).resize([224,224], resample=Image.NEAREST)
        upscaled_image = np.asarray(upscaled_image)
    return upscaled_image
def load_png_image(file_path):
    img = utils.load_img(file_path, target_size=input_shape)
    x = utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Load dataset
images = []
labels = []
image_names = []

for filename in os.listdir(target_data_dir):
    if filename.endswith('.' + image_format):
        name = filename.split('.' + image_format)[0]
        img_class = name.split('_')[1]
        labels.append(int(img_class))
        
        img_path = os.path.join(target_data_dir, filename)
        if image_format == 'png':
            image = load_png_image(img_path)
        else:
            image = load_tiff_image(img_path)
        images.append(image)
        image_names.append(filename)

labels = utils.to_categorical(labels, num_classes=2)
images = np.array(images)
labels = np.array(labels)
print(len(labels))

print('CREATE MODEL -----------------------------')

base_model = EfficientNetV2L(include_top=False, input_shape=input_shape, weights='imagenet')

for i, layer in enumerate(base_model.layers):
    layer.trainable = True

# Create a new model instance with the top layer
x = base_model.output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
predictions = tf.keras.layers.Dense(2, activation='sigmoid')(x)
model = tf.keras.Model(inputs = base_model.input, outputs = predictions)

# print(model.summary())

lr = 1e-4
optimizer = Adam(learning_rate=lr)

METRICS = [
      "accuracy",
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
      tf.keras.metrics.F1Score(threshold=0.5),
]

# Compilar modelo
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=METRICS,
)

print('EVALUATE MODEL -----------------------------')

# if using PNG file use squeeze
if image_format == 'png':
    x_all = np.squeeze(images)
else:
    x_all = images


weights_path = "./results/{}/{}.weights.h5".format(source_city, version)
model.load_weights(weights_path)

score = model.evaluate(x_all, labels, verbose=1)

print('Source: ', source_city)
print('Target: ', target_city)
print('Test loss:', score[0])
print('Test accuracy:', score[1]) 
print('Test all scores:', score) 