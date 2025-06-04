import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow.keras.utils as utils
from sklearn.model_selection import train_test_split
import json as simplejson
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import rasterio
import numpy as np
import os
from PIL import Image 
import time

print('IMPORT DATA -----------------------------')
start = time.time()


version = 'v1'
image_format = 'tiff'
target_city = 'sp'
source = 'sentinel2' # 'gmaps'
target_dataset = '{}_2024'.format(target_city.upper())
target_data_dir = '../../dataset/{}/{}'.format(source, target_dataset)

input_shape = (224, 224, 3)

# Function to load TIFF images
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

labels = utils.to_categorical(labels, num_classes=2)
images = np.array(images)
labels = np.array(labels)

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

x_train_val, x_test, y_train_val, y_test = train_test_split(images, labels, stratify=labels, test_size=test_ratio, random_state=123)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, stratify=y_train_val, test_size=val_ratio/(train_ratio+val_ratio), random_state=123)

print('CREATE MODEL -----------------------------')

base_model = EfficientNetV2L(include_top=False, input_shape=input_shape, weights='imagenet')

for i, layer in enumerate(base_model.layers):
    layer.trainable = True
    print(i, layer.name, layer.trainable)

# Create a new model instance with the top layer
x = base_model.output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
predictions = tf.keras.layers.Dense(2, activation='sigmoid')(x)
model = tf.keras.Model(inputs = base_model.input, outputs = predictions)

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

# Fit model (storing  weights) -------------------------------------------
filepath="../../dataset/results/{}/{}.weights.h5".format(target_city, version)
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, 
                             monitor='val_accuracy', 
                             verbose=1, 
                             save_best_only=True,
                             save_weights_only=True,
                             mode='max')

lr_reduce   = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, min_delta=1e-5, patience=3, verbose=0)
early       = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, mode='max')
callbacks_list = [checkpoint, lr_reduce, early]

print('TRAINING MODEL -----------------------------')

# if using PNG file use squeeze
if image_format == 'png':
    x_train = np.squeeze(x_train)
    x_val = np.squeeze(x_val)
    x_test = np.squeeze(x_test)

history = model.fit(x_train, y_train, 
          validation_data=(x_val, y_val),
          batch_size=64, 
          epochs=100, 
          verbose=1,
          callbacks=callbacks_list)

# storing Model in JSON --------------------------------------------------

model_json = model.to_json()

with open("../../dataset/results/{}/{}.json".format(target_city, version), "w") as json_file:
    json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))


### evaluate model ---------------------------------------------------------

weights_path = "../../dataset/results/{}/{}.weights.h5".format(target_city, version)
model.load_weights(weights_path)

score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1]) 
print('Test all scores:', score) 

stop = time.time()
print(f"Training time: {stop - start}s")