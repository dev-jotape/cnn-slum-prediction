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

print('IMPORT DATA -----------------------------')

version = 'sp_v1'
image_format = 'tif'

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

def load_images(target_data_dir):
    labels = []
    images = []

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

    return images, labels

# get images from all cities
source = 'sentinel2' # 'gmaps'
sp_data_dir = '../../dataset/' + source + '/SP_2024'
rj_data_dir = '../../dataset/' + source + '/RJ_2024'
bh_data_dir = '../../dataset/' + source + '/BH_2024'
br_data_dir = '../../dataset/' + source + '/BR_2024'
ssa_data_dir = '../../dataset/' + source + '/SSA_2024'
pa_data_dir = '../../dataset/' + source + '/PA_2024'

sp_x_train_val, sp_y_train_val = load_images(sp_data_dir)
rj_x_train_val, rj_y_train_val = load_images(rj_data_dir)
bh_x_train_val, bh_y_train_val = load_images(bh_data_dir)
br_x_train_val, br_y_train_val = load_images(br_data_dir)
ssa_x_train_val, ssa_y_train_val = load_images(ssa_data_dir)
pa_x_train_val, pa_y_train_val = load_images(pa_data_dir)

# Load dataset
x_train_val = np.concatenate((
    # sp_x_train_val, #remove target city
    rj_x_train_val, 
    bh_x_train_val, 
    br_x_train_val, 
    ssa_x_train_val, 
    pa_x_train_val
), axis=0)
y_train_val = np.concatenate((
    # sp_y_train_val, #remove target city
    rj_y_train_val, 
    bh_y_train_val, 
    br_y_train_val, 
    ssa_y_train_val, 
    pa_y_train_val
), axis=0)
x_test  = pa_x_train_val
y_test  = pa_y_train_val

train_ratio = 0.7
val_ratio = 0.15
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, stratify=y_train_val, test_size=val_ratio/(train_ratio+val_ratio), random_state=123)

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

# Fit model (storing  weights) -------------------------------------------
filepath="../../dataset/results/leave_one_out/{}/model.weights.h5".format(version)
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
          batch_size=32, 
          epochs=100, 
          verbose=1,
          callbacks=callbacks_list)

# storing Model in JSON --------------------------------------------------

model_json = model.to_json()

with open("../../dataset/results/leave_one_out/{}/model.json".format(version), "w") as json_file:
    json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))


### evaluate model ---------------------------------------------------------

weights_path = "../../dataset/results/leave_one_out/{}/model.weights.h5".format(version)
model.load_weights(weights_path)

score = model.evaluate(x_test, y_test, verbose=1)
print('out:', score)