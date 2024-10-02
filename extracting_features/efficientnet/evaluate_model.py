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

city = 'pa'
# dataset = 'GMAPS_RGB_2024'
# dataset = 'GEE_SENT2_RGB_2023'
# dataset = 'GEE_SENT2_RGB_2020_05'
data_dir = '../../dataset/slums_{}_images/{}/'.format(city, dataset)

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

for filename in os.listdir(data_dir):
    if filename.endswith('.tif'):
        name = filename.split('.tif')[0]
        img_class = name.split('_')[1]
        labels.append(int(img_class))
        
        img_path = os.path.join(data_dir, filename)
        # image = load_png_image(img_path)
        image = load_tiff_image(img_path)
        images.append(image)
        image_names.append(filename)

labels = utils.to_categorical(labels, num_classes=2)
images = np.array(images)
labels = np.array(labels)
print(len(labels))
# train_ratio = 0.7
# val_ratio = 0.15
# test_ratio = 0.15

# x_train_val, x_test, y_train_val, y_test = train_test_split(images, labels, stratify=labels, test_size=test_ratio, random_state=123)
# x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, stratify=y_train_val, test_size=val_ratio/(train_ratio+val_ratio), random_state=123)

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

print(model.summary())

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

print('TRAINING MODEL -----------------------------')

# if using PNG file use squeeze
x_all = np.squeeze(images)

### evaluate model ---------------------------------------------------------

weights_path = "./results/{}.weights.h5".format("GEE_SENT2_RGB_2020_05")
model.load_weights(weights_path)

score = model.evaluate(x_all, labels, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1]) 
print('Test all scores:', score) 

### Confusion Matrix -------------------------------------------------------

y_pred = model.predict(x_all)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(labels, axis=1)

# Save incorrect predictions
incorrect_predictions = []
for i in range(len(y_true)):
    if y_true[i] != y_pred_classes[i]:
        incorrect_predictions.append(image_names[i])

with open("./results/incorrect_predictions_{}_{}.txt".format(city, dataset), "w") as f:
    for item in incorrect_predictions:
        f.write("%s\n" % item)

cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)

plt.savefig("./results/confusion_matrix_evaluate_{}_{}.png".format(city, dataset))
plt.close()


# ==================== IMAGENET PRETREINED - GOOGLE MAPS ====================
# RJ
# Test loss: 0.5341697335243225
# Test accuracy: 0.8939999938011169
# Test all scores: [0.5341697335243225, 0.8939999938011169, 891.0, 105.0, 895.0, 109.0, 0.8945783376693726, 0.890999972820282, 0.9345020055770874, 0.9205309152603149, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.89374375, 0.8918098 ], dtype=float32)>]

# BH
# Test loss: 0.3148861527442932
# Test accuracy: 0.9350000023841858
# Test all scores: [0.3148861527442932, 0.9350000023841858, 936.0, 65.0, 935.0, 64.0, 0.9350649118423462, 0.9359999895095825, 0.9625355005264282, 0.9528402090072632, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.934     , 0.93706286], dtype=float32)>]

# Brasilia
# Test loss: 1.141800045967102
# Test accuracy: 0.7720000147819519
# Test all scores: [1.141800045967102, 0.7720000147819519, 774.0, 227.0, 773.0, 226.0, 0.773226797580719, 0.7739999890327454, 0.8365876078605652, 0.8098095059394836, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.8099999, 0.719101 ], dtype=float32)>]

# Salvador
# Test loss: 1.0858074426651
# Test accuracy: 0.7720000147819519
# Test all scores: [1.0858074426651, 0.7720000147819519, 770.0, 228.0, 772.0, 230.0, 0.7715430855751038, 0.7699999809265137, 0.8241145610809326, 0.7986006736755371, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.7924865, 0.7431817], dtype=float32)>]

# PA
# Test loss: 1.7684743404388428
# Test accuracy: 0.7279999852180481
# Test all scores: [1.7684743404388428, 0.7279999852180481, 727.0, 273.0, 727.0, 273.0, 0.7269999980926514, 0.7269999980926514, 0.7529335618019104, 0.7164150476455688, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.76073617, 0.6821886 ], dtype=float32)>]

# ==================== IMAGENET PRETREINED - SENTINEL-2 ====================

# RJ 
# Test loss: 1.9488403797149658
# Test accuracy: 0.7080000042915344
# Test all scores: [1.9488403797149658, 0.7080000042915344, 710.0, 297.0, 703.0, 290.0, 0.7050645351409912, 0.7099999785423279, 0.753227949142456, 0.719009518623352, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.73889863, 0.66742337], dtype=float32)>]

# BH
# Test loss: 0.90870600938797
# Test accuracy: 0.8140000104904175
# Test all scores: [0.90870600938797, 0.8140000104904175, 815.0, 185.0, 815.0, 185.0, 0.8149999976158142, 0.8149999976158142, 0.8795619606971741, 0.8593940138816833, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.79697615, 0.83053994], dtype=float32)>]

# BR
# Test loss: 0.9075028896331787
# Test accuracy: 0.8450000286102295
# Test all scores: [0.9075028896331787, 0.8450000286102295, 845.0, 155.0, 845.0, 155.0, 0.8450000286102295, 0.8450000286102295, 0.8933759927749634, 0.8709887266159058, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.8456913 , 0.84431136], dtype=float32)>]

# SSA
# Test loss: 5.411379337310791
# Test accuracy: 0.5619999766349792
# Test all scores: [5.411379337310791, 0.5619999766349792, 560.0, 438.0, 562.0, 440.0, 0.5611222386360168, 0.5600000023841858, 0.5843589901924133, 0.5646640062332153, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.6783625, 0.3047619], dtype=float32)>]

# PA
# Test loss: 2.3833789825439453
# Test accuracy: 0.6620000004768372
# Test all scores: [2.3833789825439453, 0.6620000004768372, 666.0, 340.0, 660.0, 334.0, 0.6620278358459473, 0.6660000085830688, 0.6928899884223938, 0.6582636833190918, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.7129551, 0.5939393], dtype=float32)>]