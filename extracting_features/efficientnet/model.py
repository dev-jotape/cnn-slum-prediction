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
from sklearn.metrics import roc_curve, auc

print('IMPORT DATA -----------------------------')

version = 'v1_se'
image_format = 'png'
target_city = 'sp'
target_dataset = 'GMAPS_RGB_{}_2024'.format(target_city.upper())
# dataset = 'GEE_SENT2_RGB_2020_05'
# dataset = 'GEE_SENT2_RGB_NIR_2020_05'
target_data_dir = '../../dataset/gmaps_slums/{}'.format(target_dataset)

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

totalSlum = 0
totalNonSlum = 0

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
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

x_train_val, x_test, y_train_val, y_test = train_test_split(images, labels, stratify=labels, test_size=test_ratio, random_state=123)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, stratify=y_train_val, test_size=val_ratio/(train_ratio+val_ratio), random_state=123)

print('CREATE MODEL -----------------------------')

def se_block(input_tensor, ratio=16):
    channel_axis = -1  # TensorFlow channels_last format
    filters = input_tensor.shape[channel_axis]

    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, filters))(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = multiply([input_tensor, se])
    return x

base_model = EfficientNetV2L(include_top=False, input_shape=input_shape, weights='imagenet')

for i, layer in enumerate(base_model.layers):
    layer.trainable = True
    print(i, layer.name, layer.trainable)

# Create a new model instance with the top layer
x = base_model.output
x = se_block(x)  # Adding SE block after the base model output
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

# SP c/ SE: Test all scores: [0.2524867355823517, 0.936170220375061, 1053.0, 70.0, 1058.0, 75.0, 0.9376669526100159, 0.9335106611251831, 0.9759907722473145, 0.9718819856643677, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.9346463, 0.9365078], dtype=float32)>]

# BH s/ SE: Test all scores: [0.22615084052085876, 0.9311110973358154, 415.0, 30.0, 420.0, 35.0, 0.932584285736084, 0.9222221970558167, 0.9805530309677124, 0.977677583694458, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.9244444, 0.930337 ], dtype=float32)>]

# BR s/ SE: Test all scores: [0.5444946885108948, 0.8999999761581421, 403.0, 45.0, 405.0, 47.0, 0.8995535969734192, 0.8955555558204651, 0.9437727928161621, 0.9309614896774292, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.9032257 , 0.89145494], dtype=float32)>]
# BR c/ SE: Test all scores: [0.5356412529945374, 0.8999999761581421, 400.0, 45.0, 405.0, 50.0, 0.898876428604126, 0.8888888955116272, 0.9404543042182922, 0.9299705028533936, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.8947368, 0.8929384], dtype=float32)>]

# PA s/ SE: Test all scores: [0.9034299254417419, 0.8133333325386047, 359.0, 87.0, 363.0, 91.0, 0.804932713508606, 0.7977777719497681, 0.868636965751648, 0.8490833044052124, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.79907614, 0.8034556 ], dtype=float32)>]
# PA c/ SE: Test all scores: [0.8783915638923645, 0.8111110925674438, 363.0, 86.0, 364.0, 87.0, 0.8084632754325867, 0.8066666722297668, 0.8578295707702637, 0.835511326789856, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.80182225, 0.81304336], dtype=float32)>]

# RJ s/ SE: Test all scores: [0.33343449234962463, 0.897777795791626, 395.0, 44.0, 406.0, 55.0, 0.8997722268104553, 0.8777777552604675, 0.9470025897026062, 0.94038987159729, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.87962955, 0.8971552 ], dtype=float32)>]
# RJ c/ SE: Test all scores: [0.4014453589916229, 0.9088888764381409, 411.0, 41.0, 409.0, 39.0, 0.9092920422554016, 0.9133333563804626, 0.961350679397583, 0.9555040597915649, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.90786505, 0.9146608 ], dtype=float32)>]

# SSA s/ SE: Test all scores: [0.61859130859375, 0.8533333539962769, 381.0, 64.0, 386.0, 69.0, 0.8561797738075256, 0.846666693687439, 0.9092419147491455, 0.896021842956543, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.84943813, 0.85333323], dtype=float32)>]
# SSA c/ SE: Test all scores: [0.593754768371582, 0.8355555534362793, 375.0, 73.0, 377.0, 75.0, 0.8370535969734192, 0.8333333134651184, 0.9037777781486511, 0.8933566808700562, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.8290398, 0.8407642], dtype=float32)>]

### Confusion Matrix -------------------------------------------------------

# y_pred = model.predict(x_test)
# y_pred_classes = np.argmax(y_pred, axis=1)
# y_true = np.argmax(y_test, axis=1)

# cm = confusion_matrix(y_true, y_pred_classes)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot(cmap=plt.cm.Blues)

# # Save the confusion matrix as an image file
# plt.savefig("./results/{}/confusion_matrix_{}_sp.png".format(target_city, version))
# plt.close()

### ROC curve -------------------------------------------------------

y_pred_probs = model.predict(x_test)[:, 1]  # Pega a probabilidade da classe 1
y_true = np.argmax(y_test, axis=1)

# Calcula os valores da curva ROC
fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

# Plota a curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

# Salva a curva ROC como uma imagem
roc_curve_path = "./results/{}/roc_curve_{}.png".format(target_city, version)
plt.savefig(roc_curve_path)
plt.show()
print(f"ROC curve saved to {roc_curve_path}")

# ==================== IMAGENET PRETREINED ====================
# GOOGLE MAPS
# Test loss: 
# Test accuracy:

