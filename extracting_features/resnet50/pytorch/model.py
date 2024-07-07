import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image 
from torchvision.models import resnet50, ResNet50_Weights
from torch.nn.functional import normalize, one_hot
import rasterio
import numpy as np
import sys

from train_model import PytorchTrainingAndTest

# Defina o caminho para as pastas de treinamento
data_dir = '../../../dataset/slums_sp_images/GEE_SENT2_RGB_2020_05/'

# Transformações para redimensionar e normalizar as imagens
transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize((0.0000000334, 0.000000005 , 0.000000005), (0.0000114463, 0.0000044221, 0.0000044221))
])

# Crie um conjunto de dados personalizado
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_class=2):
        self.root_dir = root_dir
        self.transform = transform
        self.num_class = num_class
        self.image_paths = []
        self.labels = []

        for image_name in os.listdir(root_dir):
            self.image_paths.append(os.path.join(root_dir, image_name))
            name = image_name.split('.tif')[0]
            img_class = name.split('_')[1]
            self.labels.append(int(img_class))
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            with rasterio.open(image_path) as src:
                image = src.read([1, 2, 3])
                image = np.moveaxis(image, 0, -1)
                upscaled_image = Image.fromarray(np.uint8(image*255)).resize([224,224], resample=Image.NEAREST)
                if self.transform:
                    upscaled_image = self.transform(upscaled_image)
            return upscaled_image, label
        except Exception as e:
            print(f"Erro ao abrir a imagem {image_path}: {e}")
            return None, None

num_class = 2

# Crie um DataLoader para o conjunto de dados
dataset = CustomDataset(data_dir, transform=transform, num_class=num_class)

dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))

# Crie os dataloaders
batch_size = 32
learning_rate = 1e-4

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for samples, targets in train_loader:
    print(samples.shape)
    print(targets.shape)
    break
# exit()

model = resnet50(weights=ResNet50_Weights.DEFAULT, progress=True)
model.feature_extractor = nn.Sequential(*list(model.children())[:-1])  # Remove the last FC layer
model.fc1 = nn.Linear(model.fc.in_features, 1024)
model.dropout = nn.Dropout(0.5)
model.fc2 = nn.Linear(1024, 1)
# model.flatten = nn.Flatten()
# model.fc1 = nn.Linear(model.fc.in_features, 1024)
# model.dropout = nn.Dropout(0.5)
# model.fc = nn.Linear(2048, 1)

# model.fc = nn.Linear(2048, 2)
# model.fc = nn.Sequential(
#     nn.Linear(model.fc.in_features, 128),
#     nn.Dropout(0.3),
#     nn.Linear(128, 1)
# )

print(model)
# exit()
trainer = PytorchTrainingAndTest()

trainer.run_model(1, model, 'Resnet50_finetuning_sent2_rgb', 'google_images', train_loader, val_loader, test_loader, learning_rate, 30, 2)