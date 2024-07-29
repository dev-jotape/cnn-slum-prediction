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


print('IMPORT DATA -----------------------------')

# Define your dataset
dataset = 'GMAPS_RGB_2024'
# dataset = 'GEE_SENT2_RGB_2020_05'
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
    if filename.endswith('.png'):
        name = filename.split('.png')[0]
        img_class = name.split('_')[1]
        labels.append(int(img_class))
        
        img_path = os.path.join(data_dir, filename)
        image = load_png_image(img_path)
        # image = load_tiff_image(img_path)
        images.append(image)

labels = utils.to_categorical(labels, num_classes=2)
images = np.array(images)
labels = np.array(labels)
print(labels)
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# x_train_val, x_test, y_train_val, y_test = train_test_split(images, labels, stratify=labels, test_size=test_ratio, random_state=123)
# x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, stratify=y_train_val, test_size=val_ratio/(train_ratio+val_ratio), random_state=123)

print('CREATE MODEL -----------------------------')

model = EfficientNetV2L(include_top=False, input_shape=input_shape, weights='imagenet')

for i, layer in enumerate(model.layers):
    layer.trainable = False
    print(i, layer.name, layer.trainable)


print(model.summary())


# if using PNG file use squeeze
X = np.squeeze(images)
print(X)
y_pred = model.predict(X)
print(y_pred)
np.save('./results/features_{}.npy'.format(dataset), y_pred)
np.save('./results/labels_{}.npy'.format(dataset), labels)


# y_pred_classes = np.argmax(y_pred, axis=1)
# y_true = np.argmax(labels, axis=1)

