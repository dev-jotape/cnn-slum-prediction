from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from tqdm.notebook import tqdm
from vit_pytorch.efficient import ViT
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
import torch.utils.data as data
import torchvision
from torchvision.transforms import ToTensor, Compose, Resize, Normalize
from torch.utils.data import DataLoader, Dataset
import os
torch.cuda.is_available()

# Hyperparameters:
batch_size = 64 
epochs = 2
lr = 3e-5
gamma = 0.7
seed = 142
IMG_SIZE = 128
patch_size = 16
num_classes = 2

transform = Compose([
    Resize((128, 128)),
    ToTensor(),
    Normalize((0.3295, 0.3599, 0.3076), (0.1815, 0.1401, 0.1258))  # Normalização
    # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalização
])

# Define your dataset
dataset = 'GMAPS_RGB_2024'
# dataset = 'GEE_SENT2_RGB_2020_05'
data_dir = '../../dataset/slums_sp_images/' + dataset + '/'

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
            name = image_name.split('.png')[0]
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

# Training device:
device = 'cuda'

# Linear Transformer:
efficient_transformer = Linformer(dim=128, seq_len=64+1, depth=12, heads=8, k=64)

# Vision Transformer Model: 
model = ViT(dim=128, image_size=128, patch_size=patch_size, num_classes=num_classes, transformer=efficient_transformer, channels=3).to(device)

# loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# Learning Rate Scheduler for Optimizer:
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

print('STARTING TRAIN')

# Training:
for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            
        for data, label in val_loader:
            
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(val_loader)
            epoch_val_loss += val_loss / len(val_loader)

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )

# Save Model:
PATH = "epochs"+"_"+str(epochs)+"_"+"img"+"_"+str(IMG_SIZE)+"_"+"patch"+"_"+str(patch_size)+"_"+"lr"+"_"+str(lr)+".pt"
torch.save(model.state_dict(), PATH)

# load saved model:
PATH = "epochs"+"_"+str(epochs)+"_"+"img"+"_"+str(IMG_SIZE)+"_"+"patch"+"_"+str(patch_size)+"_"+"lr"+"_"+str(lr)+".pt"
efficient_transformer = Linformer(dim=128, seq_len=49+1, depth=12, heads=8, k=64)
model = ViT(image_size=224, patch_size=32, num_classes=2, dim=128 ,transformer=efficient_transformer, channels=3)
model.load_state_dict(torch.load(PATH))

# Performance on Valid/Test Data
def overall_accuracy(model, test_loader, criterion):
    
    '''
    Model testing 
    
    Args:
        model: model used during training and validation
        test_loader: data loader object containing testing data
        criterion: loss function used
    
    Returns:
        test_loss: calculated loss during testing
        accuracy: calculated accuracy during testing
        y_proba: predicted class probabilities
        y_truth: ground truth of testing data
    '''
    
    y_proba = []
    y_truth = []
    test_loss = 0
    total = 0
    correct = 0
    for data in tqdm(test_loader):
        X, y = data[0].to('cpu'), data[1].to('cpu')
        output = model(X)
        test_loss += criterion(output, y.long()).item()
        for index, i in enumerate(output):
            y_proba.append(i[1])
            y_truth.append(y[index])
            if torch.argmax(i) == y[index]:
                correct+=1
            total+=1
                
    accuracy = correct/total
    
    y_proba_out = np.array([float(y_proba[i]) for i in range(len(y_proba))])
    y_truth_out = np.array([float(y_truth[i]) for i in range(len(y_truth))])
    
    return test_loss, accuracy, y_proba_out, y_truth_out


loss, acc, y_proba, y_truth = overall_accuracy(model, test_loader, criterion = nn.CrossEntropyLoss())


print(f"Accuracy: {acc}")

print(pd.value_counts(y_truth))