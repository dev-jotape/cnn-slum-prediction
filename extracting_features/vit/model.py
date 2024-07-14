import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import tensorflow.keras.utils as utils
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import rasterio
import numpy as np
import os
from PIL import Image 

import os

os.environ["KERAS_BACKEND"] = "jax"  # @param ["tensorflow", "jax", "torch"]

import keras
from keras import layers
from keras import ops

import numpy as np
import matplotlib.pyplot as plt

# import os
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'


print('IMPORT DATA -----------------------------')

# Define your dataset
dataset = 'GMAPS_RGB_2024'
# dataset = 'GEE_SENT2_RGB_2020_05'
data_dir = '../../dataset/slums_sp_images/' + dataset + '/'

input_shape = (32,32,3)
num_classes = 1

# Function to load TIFF images
def load_tiff_image(file_path):
    with rasterio.open(file_path) as src:
        image = src.read([1, 2, 3])  # Read the first three bands
        image = np.moveaxis(image, 0, -1)  # Move bands to the last dimension
        upscaled_image = Image.fromarray(np.uint8(image*255)).resize([32,32], resample=Image.NEAREST)
        upscaled_image = np.asarray(upscaled_image)
    return upscaled_image

def load_png_image(file_path):
    img = utils.load_img(file_path, target_size=input_shape)
    x = utils.img_to_array(img)
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

images = np.array(images)
# labels = utils.to_categorical(labels, num_classes=2)
labels = np.array(labels)
print(labels)
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

x_train_val, x_test, y_train_val, y_test = train_test_split(images, labels, stratify=labels, test_size=test_ratio, random_state=123)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, stratify=y_train_val, test_size=val_ratio/(train_ratio+val_ratio), random_state=123)

y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_val shape: {x_val.shape} - y_val shape: {y_val.shape}")

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 100  # For real training, use num_epochs=100. 10 is a test value
image_size = 72  # We'll resize input images to this size
patch_size = 6  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [
    2048,
    1024,
]  # Size of the dense layers of the final classifier

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(x_train)

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

plt.figure(figsize=(4, 4))
image = x_train[np.random.choice(range(x_train.shape[0]))]
plt.imshow(image.astype("uint8"))
plt.axis("off")
plt.savefig('teste1.png')

resized_image = ops.image.resize(
    ops.convert_to_tensor([image]), size=(image_size, image_size)
)
patches = Patches(patch_size)(resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = ops.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(ops.convert_to_numpy(patch_img).astype("uint8"))
    plt.axis("off")
plt.savefig('teste2.png')

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config
    
def create_vit_classifier():
    inputs = keras.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

def run_experiment(model):
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    # model.compile(
    #     optimizer=optimizer,
    #     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #     metrics=[
    #         keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
    #         keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
    #     ],
    # )
    METRICS = [
      "accuracy",
    #   tf.keras.metrics.TruePositives(name='tp'),
    #   tf.keras.metrics.FalsePositives(name='fp'),
    #   tf.keras.metrics.TrueNegatives(name='tn'),
    #   tf.keras.metrics.FalseNegatives(name='fn'), 
    #   tf.keras.metrics.Precision(name='precision'),
    #   tf.keras.metrics.Recall(name='recall'),
    #   tf.keras.metrics.AUC(name='auc'),
    #   tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
      tf.keras.metrics.F1Score(threshold=0.5, name='f1'),
    ]

    # Compilar modelo
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=METRICS,
    )

    checkpoint_filepath = "/tmp/checkpoint.weights.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )
    early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, mode='max')

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback, early],
    )

    model.load_weights(checkpoint_filepath)
    metrics = model.evaluate(x_test, y_test)
    # print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    # print(f"Test f1: {round(f1 * 100, 2)}%")
    print(f"Test metrics: {metrics}%")

    return history


vit_classifier = create_vit_classifier()
history = run_experiment(vit_classifier)


# def plot_history(item):
#     plt.plot(history.history[item], label=item)
#     plt.plot(history.history["val_" + item], label="val_" + item)
#     plt.xlabel("Epochs")
#     plt.ylabel(item)
#     plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
#     plt.legend()
#     plt.grid()
#     plt.show()
#     plt.savefig('history.png')


# plot_history("loss")
# plot_history("top-5-accuracy")