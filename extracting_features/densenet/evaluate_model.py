import tensorflow as tf
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow.keras.utils as utils
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import rasterio
import numpy as np
import os
from PIL import Image 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input


print('IMPORT DATA -----------------------------')

city = 'pa'
dataset = 'GMAPS_RGB_2024'
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
    if filename.endswith('.png'):
        name = filename.split('.png')[0]
        img_class = name.split('_')[1]
        labels.append(int(img_class))
        
        img_path = os.path.join(data_dir, filename)
        image = load_png_image(img_path)
        # image = load_tiff_image(img_path)
        images.append(image)
        image_names.append(filename)

labels = utils.to_categorical(labels, num_classes=2)
images = np.array(images)
labels = np.array(labels)
print(len(labels))

print('CREATE MODEL -----------------------------')

base_model = DenseNet201(include_top=False, input_shape=input_shape, weights='imagenet')

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

weights_path = "./results/{}.weights.h5".format(dataset)
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

# ==================== IMAGENET PRETREINED - SENTINEL-2 ====================

# RJ 
