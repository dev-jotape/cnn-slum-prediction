import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D, GlobalAveragePooling2D, Dense, Input, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet201
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
from tensorflow.keras.applications.resnet50 import preprocess_input

print('IMPORT DATA -----------------------------')

# Define your dataset
# dataset = 'GMAPS_RGB_2024'
dataset = 'GEE_SENT2_RGB_2020_05'
# dataset = 'GEE_SENT2_RGB_NIR_2020_05'
data_dir = '../../dataset/slums_sp_images/' + dataset + '/'

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

for filename in os.listdir(data_dir):
    if filename.endswith('.tif'):
        name = filename.split('.tif')[0]
        img_class = name.split('_')[1]
        labels.append(int(img_class))
        
        img_path = os.path.join(data_dir, filename)
        # image = load_png_image(img_path)
        image = load_tiff_image(img_path)
        images.append(image)

labels = utils.to_categorical(labels, num_classes=2)
images = np.array(images)
labels = np.array(labels)
print(labels)
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

x_train_val, x_test, y_train_val, y_test = train_test_split(images, labels, stratify=labels, test_size=test_ratio, random_state=123)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, stratify=y_train_val, test_size=val_ratio/(train_ratio+val_ratio), random_state=123)

# if using PNG file use squeeze
x_train = np.squeeze(x_train)
x_val = np.squeeze(x_val)
x_test = np.squeeze(x_test)

print('CREATE MODEL -----------------------------')

def squeeze_excite_block(input, filters):
    se = tf.keras.layers.GlobalAveragePooling2D()(input)
    se = tf.keras.layers.Reshape((1, 1, filters))(se)
    se = tf.keras.layers.Dense(filters // 16, activation='relu')(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid')(se)
    return tf.keras.layers.multiply([input, se])

input_layer = Input(shape=(None, None, 3))

# # Initial Convolutions
# conv1 = Conv2D(16, (5, 5), padding='same')(input_layer)
# conv2 = Conv2D(32, (3, 3), padding='same')(input_layer)
# conv3 = Conv2D(16, (1, 1), padding='same')(input_layer)

# # Concatenation
# concat = tf.keras.layers.Concatenate()([conv1, conv2, conv3])

# # Convolutional Block 1
# conv_block1 = Conv2D(64, (3, 3), padding='same')(concat)
# conv_block1 = BatchNormalization()(conv_block1)
# conv_block1 = ReLU()(conv_block1)
# conv_block1 = squeeze_excite_block(conv_block1, 64)

# # Convolutional Block 2
# conv_block2 = Conv2D(64, (3, 3), padding='same')(conv_block1)
# conv_block2 = BatchNormalization()(conv_block2)
# conv_block2 = ReLU()(conv_block2)
# conv_block2 = squeeze_excite_block(conv_block2, 64)

# # Convolutional Block 3
# conv_block3 = Conv2D(128, (3, 3), padding='same')(conv_block2)
# conv_block3 = BatchNormalization()(conv_block3)
# conv_block3 = ReLU()(conv_block3)
# conv_block3 = squeeze_excite_block(conv_block3, 128)

# # MaxPooling and Convolutional Block 4
# maxpool = MaxPooling2D(pool_size=(1, 1))(conv_block3)
# conv_block4 = Conv2D(128, (3, 3), padding='same')(maxpool)
# conv_block4 = BatchNormalization()(conv_block4)
# conv_block4 = ReLU()(conv_block4)
# conv_block4 = squeeze_excite_block(conv_block4, 128)

# # Convolutional Block 5
# conv_block5 = Conv2D(256, (3, 3), padding='same')(conv_block4)
# conv_block5 = BatchNormalization()(conv_block5)
# conv_block5 = ReLU()(conv_block5)
# conv_block5 = squeeze_excite_block(conv_block5, 256)

# # Convolutional Block 6
# conv_block6 = Conv2D(256, (3, 3), padding='same')(conv_block5)
# conv_block6 = BatchNormalization()(conv_block6)
# conv_block6 = ReLU()(conv_block6)
# conv_block6 = squeeze_excite_block(conv_block6, 256)

# # Global Average Pooling and Output
# gap = GlobalAveragePooling2D()(conv_block6)
# output = Dense(2, activation='softmax')(gap)  # Adjust the number of classes as needed

# First Convolutional Block
x = Conv2D(8, (3, 3), padding='same', activation='relu')(input_layer)
x = BatchNormalization()(x)
x = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

# Second Convolutional Block
x = Conv2D(16, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(16, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

# Third Convolutional Block
x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

# Fourth Convolutional Block
x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

# Fifth Convolutional Block
x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

# Sixth Convolutional Block
x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

# Fully Connected Layer
x = Flatten()(x)
x = Dense(196, activation='relu')(x)
x = Dropout(0.4, seed=1234)(x)
output = Dense(2, activation='softmax')(x)

# Model
model = Model(inputs=input_layer, outputs=output)

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

model.compile(
    loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=METRICS,
)

# Fit model (storing  weights) -------------------------------------------
filepath="./results/{}.weights.h5".format(dataset)
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

history = model.fit(
    x_train, y_train, 
    batch_size=32, 
    validation_data=(x_val, y_val),
    epochs=100, 
    verbose=1,
    callbacks=callbacks_list)

score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1]) 
print('Test all scores:', score) 