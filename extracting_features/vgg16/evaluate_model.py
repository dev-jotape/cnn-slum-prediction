import tensorflow as tf
from tensorflow.keras.applications import VGG16
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
# dataset = 'GMAPS_RGB_2024'
# dataset = 'GEE_SENT2_RGB_2020_05'
dataset = 'GEE_SENT2_RGB_2023'
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

print('CREATE MODEL -----------------------------')

base_model = VGG16(include_top=False, input_shape=input_shape)

for i, layer in enumerate(base_model.layers):
    layer.trainable = False
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
# x_all = images

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
# Test loss: 0.3459961414337158
# Test accuracy: 0.8799999952316284
# Test all scores: [0.3459961414337158, 0.8799999952316284, 879.0, 122.0, 878.0, 121.0, 0.8781218528747559, 0.8790000081062317, 0.9451839923858643, 0.9408930540084839, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.87750995, 0.87960196], dtype=float32)>]

# BH
# Test loss: 0.21308088302612305
# Test accuracy: 0.925000011920929
# Test all scores: [0.21308088302612305, 0.925000011920929, 925.0, 76.0, 924.0, 75.0, 0.9240759015083313, 0.925000011920929, 0.9743870496749878, 0.9717444777488708, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.9241657, 0.9249011], dtype=float32)>]

# BR
# Test loss: 0.7075560092926025
# Test accuracy: 0.7820000052452087
# Test all scores: [0.7075560092926025, 0.7820000052452087, 786.0, 219.0, 781.0, 214.0, 0.7820895314216614, 0.7860000133514404, 0.8391099572181702, 0.8160444498062134, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.8112723, 0.7458033], dtype=float32)>]

# SSA
# Test loss: 1.1375079154968262
# Test accuracy: 0.7089999914169312
# Test all scores: [1.1375079154968262, 0.7089999914169312, 707.0, 290.0, 710.0, 293.0, 0.7091273665428162, 0.7070000171661377, 0.7461369633674622, 0.7201516032218933, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.754653 , 0.6404908], dtype=float32)>]

# PA
# Test loss: 1.1293940544128418
# Test accuracy: 0.6840000152587891
# Test all scores: [1.1293940544128418, 0.6840000152587891, 684.0, 316.0, 684.0, 316.0, 0.6840000152587891, 0.6840000152587891, 0.7368835210800171, 0.7119393348693848, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.7224669 , 0.63352597], dtype=float32)>]

# ==================== IMAGENET PRETREINED - SENTINEL-2 ====================

# RJ
# Test loss: 0.9425138235092163
# Test accuracy: 0.7039999961853027
# Test all scores: [0.9425138235092163, 0.7039999961853027, 706.0, 293.0, 707.0, 294.0, 0.706706702709198, 0.7059999704360962, 0.7540804743766785, 0.7120596170425415, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.7080867, 0.7045685], dtype=float32)>]

# BH
# Test loss: 0.472760945558548
# Test accuracy: 0.8180000185966492
# Test all scores: [0.472760945558548, 0.8180000185966492, 818.0, 181.0, 819.0, 182.0, 0.8188188076019287, 0.8180000185966492, 0.8911409378051758, 0.8892968893051147, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.8072417, 0.8283018], dtype=float32)>]

# BR
# Test loss: 0.6268860697746277
# Test accuracy: 0.8399999737739563
# Test all scores: [0.6268860697746277, 0.8399999737739563, 842.0, 158.0, 842.0, 158.0, 0.8420000076293945, 0.8420000076293945, 0.8968909978866577, 0.874847948551178, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.84304035, 0.840932  ], dtype=float32)>]

# SSA
# Test loss: 1.9465261697769165
# Test accuracy: 0.5899999737739563
# Test all scores: [1.9465261697769165, 0.5899999737739563, 589.0, 408.0, 592.0, 411.0, 0.5907723307609558, 0.5889999866485596, 0.5915155410766602, 0.565049409866333, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.6725239 , 0.45100662], dtype=float32)>]

# PA
# Test loss: 1.2240337133407593
# Test accuracy: 0.6819999814033508
# Test all scores: [1.2240337133407593, 0.6819999814033508, 684.0, 317.0, 683.0, 316.0, 0.683316707611084, 0.6840000152587891, 0.7168784