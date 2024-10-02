import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
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
data_dir = '../../../dataset/slums_{}_images/{}/'.format(city, dataset)

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

base_model = ResNet50(include_top=False, input_shape=input_shape)

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
# Test loss: 0.8567430973052979
# Test accuracy: 0.8569999933242798
# Test all scores: [0.8567430973052979, 0.8569999933242798, 859.0, 145.0, 855.0, 141.0, 0.8555777072906494, 0.859000027179718, 0.9152384996414185, 0.8986262083053589, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.8519269 , 0.86247545], dtype=float32)>]

# BH
# Test loss: 0.49826982617378235
# Test accuracy: 0.9259999990463257
# Test all scores: [0.49826982617378235, 0.9259999990463257, 927.0, 77.0, 923.0, 73.0, 0.9233067631721497, 0.9269999861717224, 0.9634705185890198, 0.9547291398048401, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.92673266, 0.9235412 ], dtype=float32)>]

# BR
# Test loss: 1.2612475156784058
# Test accuracy: 0.8159999847412109
# Test all scores: [1.2612475156784058, 0.8159999847412109, 821.0, 181.0, 819.0, 179.0, 0.8193612694740295, 0.8209999799728394, 0.8701614737510681, 0.8431018590927124, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.8433946 , 0.78928983], dtype=float32)>]

# SSA
# Test loss: 1.781467318534851
# Test accuracy: 0.7300000190734863
# Test all scores: [1.781467318534851, 0.7300000190734863, 730.0, 276.0, 724.0, 270.0, 0.7256461381912231, 0.7300000190734863, 0.7801920175552368, 0.7484380006790161, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.7702702 , 0.66666657], dtype=float32)>]

# PA
# Test loss: 2.4797050952911377
# Test accuracy: 0.6850000023841858
# Test all scores: [2.4797050952911377, 0.6850000023841858, 686.0, 321.0, 679.0, 314.0, 0.6812313795089722, 0.6859999895095825, 0.6985085010528564, 0.6628377437591553, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.732773  , 0.61199504], dtype=float32)>]

# ==================== IMAGENET PRETREINED - SENTINEL-2 ====================

# RJ 
# Test loss: 3.948249101638794
# Test accuracy: 0.6389999985694885
# Test all scores: [3.948249101638794, 0.6389999985694885, 646.0, 374.0, 626.0, 354.0, 0.6333333253860474, 0.6460000276565552, 0.6614595055580139, 0.629518449306488, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.7161437 , 0.49929872], dtype=float32)>]

# BH
# Test loss: 0.8167446255683899
# Test accuracy: 0.796999990940094
# Test all scores: [0.8167446255683899, 0.796999990940094, 798.0, 207.0, 793.0, 202.0, 0.7940298318862915, 0.7979999780654907, 0.8698664903640747, 0.8563234806060791, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.77753294, 0.8113035 ], dtype=float32)>]

# BR
# Test loss: 1.0858465433120728
# Test accuracy: 0.8100000023841858
# Test all scores: [1.0858465433120728, 0.8100000023841858, 808.0, 189.0, 811.0, 192.0, 0.8104313015937805, 0.8080000281333923, 0.8635525107383728, 0.8388285636901855, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.81481475, 0.8032955 ], dtype=float32)>]

# SSA
# Test loss: 6.289668083190918
# Test accuracy: 0.531000018119812
# Test all scores: [6.289668083190918, 0.531000018119812, 534.0, 473.0, 527.0, 466.0, 0.5302879810333252, 0.5339999794960022, 0.5495569705963135, 0.5368368625640869, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.67222214, 0.1763668 ], dtype=float32)>]

# PA
# Test loss: 3.4346861839294434
# Test accuracy: 0.6159999966621399
# Test all scores: [3.4346861839294434, 0.6159999966621399, 620.0, 388.0, 612.0, 380.0, 0.6150793433189392, 0.6200000047683716, 0.6391685009002686, 0.6103368997573853, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.70991796, 0.43178403], dtype=float32)>]