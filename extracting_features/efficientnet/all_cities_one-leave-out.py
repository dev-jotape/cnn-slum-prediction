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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply
from tensorflow.keras.applications.resnet50 import preprocess_input
import datetime

print('IMPORT DATA -----------------------------')

version = 'pa_sentinel'
image_format = 'tif'
target_city = 'leave_one_out'

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

def load_images(target_data_dir):
    labels = []
    images = []

    totalSlum = 0
    totalNonSlum = 0

    for filename in os.listdir(target_data_dir):
        if filename.endswith('.' + image_format):
            name = filename.split('.' + image_format)[0]
            img_class = name.split('_')[1]

            if img_class == '1':
                totalSlum = totalSlum + 1
                if totalSlum > 1500:
                    continue

            if img_class == '0':
                totalNonSlum = totalNonSlum + 1
                if totalNonSlum > 1500:
                    continue 

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
    print('size: ', len(labels))
    # test_ratio = 0.15
    # x_train_val, x_test, y_train_val, y_test = train_test_split(images, labels, stratify=labels, test_size=test_ratio, random_state=123)
    # return x_train_val, x_test, y_train_val, y_test
    return images, labels

# get images from all cities
sp_data_dir = '../../dataset/sentinel2_slums/GEE_SENT2_RGB_SP_2024'
rj_data_dir = '../../dataset/sentinel2_slums/GEE_SENT2_RGB_RJ_2024'
bh_data_dir = '../../dataset/sentinel2_slums/GEE_SENT2_RGB_BH_2024'
br_data_dir = '../../dataset/sentinel2_slums/GEE_SENT2_RGB_BR_2024'
ssa_data_dir = '../../dataset/sentinel2_slums/GEE_SENT2_RGB_SSA_2024'
pa_data_dir = '../../dataset/sentinel2_slums/GEE_SENT2_RGB_PA_2024'

# sp_data_dir = '../../dataset/gmaps_slums/GMAPS_RGB_SP_2024'
# rj_data_dir = '../../dataset/gmaps_slums/GMAPS_RGB_RJ_2024'
# bh_data_dir = '../../dataset/gmaps_slums/GMAPS_RGB_BH_2024'
# br_data_dir = '../../dataset/gmaps_slums/GMAPS_RGB_BR_2024'
# ssa_data_dir = '../../dataset/gmaps_slums/GMAPS_RGB_SSA_2024'
# pa_data_dir = '../../dataset/gmaps_slums/GMAPS_RGB_PA_2024'

sp_x_train_val, sp_y_train_val = load_images(sp_data_dir)
rj_x_train_val, rj_y_train_val = load_images(rj_data_dir)
bh_x_train_val, bh_y_train_val = load_images(bh_data_dir)
br_x_train_val, br_y_train_val = load_images(br_data_dir)
ssa_x_train_val, ssa_y_train_val = load_images(ssa_data_dir)
pa_x_train_val, pa_y_train_val = load_images(pa_data_dir)

# Load dataset
x_train_val = np.concatenate((
    sp_x_train_val, 
    rj_x_train_val, 
    bh_x_train_val, 
    br_x_train_val, 
    ssa_x_train_val, 
    # pa_x_train_val
), axis=0)
y_train_val = np.concatenate((
    sp_y_train_val, 
    rj_y_train_val, 
    bh_y_train_val, 
    br_y_train_val, 
    ssa_y_train_val, 
    # pa_y_train_val
), axis=0)
x_test  = pa_x_train_val
y_test  = pa_y_train_val

train_ratio = 0.7
val_ratio = 0.15
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, stratify=y_train_val, test_size=val_ratio/(train_ratio+val_ratio), random_state=123)

print(len(x_train), len(y_train))
print(len(x_val), len(y_val))
print(len(x_test), len(y_test))

print('CREATE MODEL -----------------------------')

# def se_block(input_tensor, ratio=16):
#     channel_axis = -1  # TensorFlow channels_last format
#     filters = input_tensor.shape[channel_axis]

#     se = GlobalAveragePooling2D()(input_tensor)
#     se = Reshape((1, 1, filters))(se)
#     se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
#     se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
#     x = multiply([input_tensor, se])
#     return x

base_model = EfficientNetV2L(include_top=False, input_shape=input_shape, weights='imagenet')

for i, layer in enumerate(base_model.layers):
    layer.trainable = True

# Create a new model instance with the top layer
x = base_model.output
# x = se_block(x)  # Adding SE block after the base model output
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
filepath="../../dataset/results/{}/{}/model.weights.h5".format(target_city, version)
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
    # sp_x_test = np.squeeze(sp_x_test)
    # rj_x_test = np.squeeze(rj_x_test)
    # bh_x_test = np.squeeze(bh_x_test)
    # br_x_test = np.squeeze(br_x_test)
    # ssa_x_test = np.squeeze(ssa_x_test)
    # pa_x_test = np.squeeze(pa_x_test)

history = model.fit(x_train, y_train, 
          validation_data=(x_val, y_val),
          batch_size=32, 
          epochs=100, 
          verbose=1,
          callbacks=callbacks_list)

# storing Model in JSON --------------------------------------------------

model_json = model.to_json()

with open("../../dataset/results/{}/{}/model.json".format(target_city, version), "w") as json_file:
    json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))


### evaluate model ---------------------------------------------------------

weights_path = "../../dataset/results/{}/{}/model.weights.h5".format(target_city, version)
model.load_weights(weights_path)

score = model.evaluate(x_test, y_test, verbose=1)
print('out:', score) 


# GOOGLE MAPS
# PA out: [3.98095440864563, 0.7319999933242798, 2198.0, 802.0, 2198.0, 802.0, 0.7326666712760925, 0.7326666712760925, 0.752567708492279, 0.7146792411804199, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.7704636 , 0.67996806], dtype=float32)>]
# SP out: [0.7099341750144958, 0.8629999756813049, 2593.0, 410.0, 2590.0, 407.0, 0.8634698390960693, 0.8643333315849304, 0.919573962688446, 0.903495192527771, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.8684464, 0.8590186], dtype=float32)>]
# RJ out: [0.9458364844322205, 0.8733333349227905, 2614.0, 379.0, 2621.0, 386.0, 0.8733711838722229, 0.8713333606719971, 0.9166488647460938, 0.8991209864616394, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.87038916, 0.87425935], dtype=float32)>]
# BH out: [0.486162394285202, 0.8999999761581421, 2700.0, 300.0, 2700.0, 300.0, 0.8999999761581421, 0.8999999761581421, 0.9493570327758789, 0.9390371441841125, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.89970005, 0.9003    ], dtype=float32)>]
# SSA out: [0.8703770637512207, 0.809333324432373, 2431.0, 578.0, 2422.0, 569.0, 0.8079096078872681, 0.8103333115577698, 0.8681621551513672, 0.8474045395851135, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.811221 , 0.8069821], dtype=float32)>]
# BR out: [1.4784529209136963, 0.7596666812896729, 2271.0, 721.0, 2279.0, 729.0, 0.7590240836143494, 0.7570000290870667, 0.8108009099960327, 0.7816914916038513, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.7994459 , 0.69521403], dtype=float32)>]

# SENTINEL
# SP out: [1.1457345485687256, 0.7443333268165588, 2231.0, 764.0, 2236.0, 769.0, 0.7449081540107727, 0.7436666488647461, 0.8126908540725708, 0.7893695831298828, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.7792507 , 0.69623756], dtype=float32)>]
# RJ out: [1.3577138185501099, 0.7733333110809326, 2322.0, 678.0, 2322.0, 678.0, 0.7739999890327454, 0.7739999890327454, 0.8295598030090332, 0.8015890121459961, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.7511012, 0.7930402], dtype=float32)>]
# BH out: [0.8939681053161621, 0.7419999837875366, 2253.0, 803.0, 2197.0, 747.0, 0.7372382283210754, 0.7509999871253967, 0.8097867369651794, 0.7875454425811768, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.745317 , 0.7427813], dtype=float32)>]
# BR out: [1.4642854928970337, 0.7583333253860474, 2273.0, 727.0, 2273.0, 727.0, 0.7576666474342346, 0.7576666474342346, 0.8112688660621643, 0.7811938524246216, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.7871157 , 0.71876204], dtype=float32)>]
# SSA out: [2.492009162902832, 0.6330000162124634, 1896.0, 1098.0, 1902.0, 1104.0, 0.6332665085792542, 0.6320000290870667, 0.6770798563957214, 0.6478960514068604, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.67932236, 0.57042795], dtype=float32)>]
# PA out: [1.1787246465682983, 0.6623333096504211, 2000.0, 1028.0, 1972.0, 1000.0, 0.6605019569396973, 0.6666666865348816, 0.6726547479629517, 0.6231554746627808, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.69940823, 0.6178247 ], dtype=float32)>]