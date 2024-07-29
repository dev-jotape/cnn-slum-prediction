import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import timm
from lightning.pytorch import Trainer
import tempfile

from torchgeo.trainers import ClassificationTask
from torchgeo.models import ResNet18_Weights
from train_model import PytorchTrainingAndTest

# Defina o caminho para as pastas de treinamento
dataset = 'GMAPS_RGB_2024'
# dataset = 'GEE_SENT2_RGB_2020_05'
# dataset = 'GEE_SENT2_RGB_NIR_2020_05'
data_dir = '../../dataset/slums_sp_images/' + dataset + '/'

# Transformações para redimensionar e normalizar as imagens
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalização
])

# Crie um conjunto de dados personalizado
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for image_name in os.listdir(root_dir):
            self.image_paths.append(os.path.join(root_dir, image_name))
            name = image_name.split('.')[0]
            img_class = name.split('_')[1]
            self.labels.append(int(img_class))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

# Crie um DataLoader para o conjunto de dados
dataset = CustomDataset(data_dir, transform=transform)
dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

print(dataset_size)
print(train_size+val_size+test_size)
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Crie os dataloaders
# batch_size = 10
num_workers = 2
max_epochs = 10
fast_dev_run = False
batch_size = 32
learning_rate = 1e-4

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

weights = ResNet18_Weights.SENTINEL2_ALL_MOCO
in_chans = weights.meta["in_chans"]

model = timm.create_model("resnet18", in_chans=in_chans, num_classes=2)
model.load_state_dict(weights.get_state_dict(progress=True), strict=False)

accelerator = "gpu" if torch.cuda.is_available() else "cpu"
default_root_dir = os.path.join(tempfile.gettempdir(), "experiments")

# trainer = Trainer(
#     accelerator=accelerator,
#     default_root_dir=default_root_dir,
#     fast_dev_run=fast_dev_run,
#     log_every_n_steps=1,
#     min_epochs=1,
#     max_epochs=max_epochs,
# )
# trainer.fit(model=task, train_dataloaders=train_loader)
trainer = PytorchTrainingAndTest()
trainer.run_model(1, model, 'Inception3_finetuning_gmm3', 'google_images', train_loader, val_loader, test_loader, learning_rate, 1, 2)