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

version = 'v2'
image_format = 'png'
target_city = 'all'

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
    print(len(labels))
    test_ratio = 0.15
    x_train_val, x_test, y_train_val, y_test = train_test_split(images, labels, stratify=labels, test_size=test_ratio, random_state=123)
    return x_train_val, x_test, y_train_val, y_test

# get images from all cities
sp_data_dir = '../../dataset/gmaps_slums/GMAPS_RGB_SP_2024'
rj_data_dir = '../../dataset/gmaps_slums/GMAPS_RGB_RJ_2024'
bh_data_dir = '../../dataset/gmaps_slums/GMAPS_RGB_BH_2024'
br_data_dir = '../../dataset/gmaps_slums/GMAPS_RGB_BR_2024'
ssa_data_dir = '../../dataset/gmaps_slums/GMAPS_RGB_SSA_2024'
pa_data_dir = '../../dataset/gmaps_slums/GMAPS_RGB_PA_2024'

sp_x_train_val, sp_x_test, sp_y_train_val, sp_y_test = load_images(sp_data_dir)
rj_x_train_val, rj_x_test, rj_y_train_val, rj_y_test = load_images(rj_data_dir)
bh_x_train_val, bh_x_test, bh_y_train_val, bh_y_test = load_images(bh_data_dir)
br_x_train_val, br_x_test, br_y_train_val, br_y_test = load_images(br_data_dir)
ssa_x_train_val, ssa_x_test, ssa_y_train_val, ssa_y_test = load_images(ssa_data_dir)
pa_x_train_val, pa_x_test, pa_y_train_val, pa_y_test = load_images(pa_data_dir)

# Load dataset
x_train_val = np.concatenate((sp_x_train_val, rj_x_train_val, bh_x_train_val, br_x_train_val, ssa_x_train_val, pa_x_train_val), axis=0)
y_train_val = np.concatenate((sp_y_train_val, rj_y_train_val, bh_y_train_val, br_y_train_val, ssa_y_train_val, pa_y_train_val), axis=0)
x_test  = np.concatenate((sp_x_test, rj_x_test, bh_x_test, br_x_test, ssa_x_test, pa_x_test), axis=0)
y_test  = np.concatenate((sp_y_test, rj_y_test, bh_y_test, br_y_test, ssa_y_test, pa_y_test), axis=0)

train_ratio = 0.7
val_ratio = 0.15
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, stratify=y_train_val, test_size=val_ratio/(train_ratio+val_ratio), random_state=123)

print(len(x_train), len(y_train))
print(len(x_val), len(y_val))
print(len(x_test), len(y_test))
print("SP: ", len(sp_x_test))
print("PA: ", len(pa_x_test))

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
# filepath="../../dataset/results/{}/{}/model.weights.h5".format(target_city, version)
# checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, 
#                              monitor='val_accuracy', 
#                              verbose=1, 
#                              save_best_only=True,
#                              save_weights_only=True,
#                              mode='max')

# lr_reduce   = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, min_delta=1e-5, patience=3, verbose=0)
# early       = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, mode='max')
# callbacks_list = [checkpoint, lr_reduce, early]

print('TRAINING MODEL -----------------------------')

# if using PNG file use squeeze
if image_format == 'png':
    x_train = np.squeeze(x_train)
    x_val = np.squeeze(x_val)
    x_test = np.squeeze(x_test)
    sp_x_test = np.squeeze(sp_x_test)
    rj_x_test = np.squeeze(rj_x_test)
    bh_x_test = np.squeeze(bh_x_test)
    br_x_test = np.squeeze(br_x_test)
    ssa_x_test = np.squeeze(ssa_x_test)
    pa_x_test = np.squeeze(pa_x_test)

# history = model.fit(x_train, y_train, 
#           validation_data=(x_val, y_val),
#           batch_size=32, 
#           epochs=100, 
#           verbose=1,
#           callbacks=callbacks_list)

# storing Model in JSON --------------------------------------------------

model_json = model.to_json()

# with open("../../dataset/results/{}/{}/model.json".format(target_city, version), "w") as json_file:
#     json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))


### evaluate model ---------------------------------------------------------

weights_path = "../../dataset/results/{}/{}/model.weights.h5".format(target_city, version)
model.load_weights(weights_path)

score_sp = model.evaluate(sp_x_test, sp_y_test, verbose=1)
score_rj = model.evaluate(rj_x_test, rj_y_test, verbose=1)
score_bh = model.evaluate(bh_x_test, bh_y_test, verbose=1)
score_br = model.evaluate(br_x_test, br_y_test, verbose=1)
score_ssa = model.evaluate(ssa_x_test, ssa_y_test, verbose=1)
score_pa = model.evaluate(pa_x_test, pa_y_test, verbose=1)
print('sp:', score_sp) 
print('rj:', score_rj) 
print('bh:', score_bh) 
print('br:', score_br) 
print('ssa:', score_ssa) 
print('pa:', score_pa)

# sp:	0.538634717464447	0.9116666913032532	546.0	53.0	547.0	54.0	0.9115191698074341	0.9100000262260437	0.9473389983177185	0.9328979849815369	0.91089106
# rj:	0.7101267576217651	0.9066666960716248	544.0	58.0	542.0	56.0	0.9036544561386108	0.9066666960716248	0.9426820278167725	0.9268550872802734	0.9090908
# bh:	0.2759113609790802	0.9449999928474426	567.0	32.0	568.0	33.0	0.9465776085853577	0.9449999928474426	0.9768652319908142	0.9714370369911194	0.9478827
# br:	0.6250314116477966	0.9016666412353516	543.0	60.0	540.0	57.0	0.9004974961280823	0.9049999713897705	0.9383319020271301	0.9217382669448853	0.9053627
# ssa:	0.8963808417320251	0.8700000047683716	522.0	80.0	520.0	78.0	0.8671096563339233	0.8700000047683716	0.913016676902771	0.8929194211959839	0.8721311
# pa:	1.0889812707901	    0.8266666531562805	496.0	106.0	494.0	104.0	0.8239202499389648	0.8266666531562805	0.8771235346794128	0.8555381298065186	0.8268907

# sp: [0.40004023909568787, 0.9133333563804626, 411.0, 40.0, 410.0, 39.0, 0.911308228969574, 0.9133333563804626, 0.9607234597206116, 0.9547058939933777, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.9162995, 0.9082774], dtype=float32)>]
# rj: [0.45276638865470886, 0.8999999761581421, 405.0, 45.0, 405.0, 45.0, 0.8999999761581421, 0.8999999761581421, 0.9475902318954468, 0.938201904296875, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.8955916 , 0.90405107], dtype=float32)>]
# bh: [0.23802752792835236, 0.9155555367469788, 411.0, 38.0, 412.0, 39.0, 0.9153674840927124, 0.9133333563804626, 0.9755531549453735, 0.9735321998596191, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.9115646 , 0.91703045], dtype=float32)>]
# br: [0.3697082996368408, 0.8999999761581421, 404.0, 45.0, 405.0, 46.0, 0.8997772932052612, 0.897777795791626, 0.9602074027061462, 0.9519544839859009, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.90280765, 0.8944953 ], dtype=float32)>]
# ssa: [0.6606684923171997, 0.8577777743339539, 386.0, 62.0, 388.0, 64.0, 0.8616071343421936, 0.8577777743339539, 0.9109036922454834, 0.897025465965271, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.8571428 , 0.86214435], dtype=float32)>]
# pa: [0.7740493416786194, 0.7888888716697693, 351.0, 93.0, 357.0, 99.0, 0.7905405163764954, 0.7799999713897705, 0.8684889078140259, 0.8556915521621704, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.77777773, 0.7922078 ], dtype=float32)>]