import torch
from torch.utils.data import DataLoader, Dataset
import os
from torch.nn.functional import normalize
from PIL import Image 
import rasterio
from torchvision import transforms
import numpy as np

# data_dir = '../dataset/slums_sp_images/GEE_SENT2_RGB_2020_05/'
data_dir = '../dataset/slums_sp_images/GEE_SENT2_RGB_2020_05/'

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
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
        try:
            with rasterio.open(image_path) as src:
                image = src.read([1, 2, 3])
                image = np.moveaxis(image, 0, -1)
                image = image.astype(np.uint8)

            image = Image.fromarray(image)
            
            if self.transform:
                image = self.transform(image)
                
            label = self.labels[idx]
            return image, label
        except Exception as e:
            print(f"Erro ao abrir a imagem {image_path}: {e}")
            return None, None

dataset = CustomDataset(data_dir, transform=transform)

dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size
batch_size = 32

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

channels_sum, channels_squared_sum, num_batches = 0, 0, 0
for i, sample in enumerate(train_loader):
    img = sample[0]
    channels_sum += torch.mean(img, dim=[0, 2, 3])
    channels_squared_sum += torch.mean(img**2, dim=[0, 2, 3])
    num_batches += 1

mean = channels_sum / num_batches
std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

norm_params = {"mean": mean.numpy(), "std": std.numpy()}
np.set_printoptions(suppress=True, precision=10)
print(norm_params)
