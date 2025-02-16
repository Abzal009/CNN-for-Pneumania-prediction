import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import zipfile
import matplotlib.pyplot as plt
import pickle

class ImageDataset(Dataset):
    def __init__(self, zip_path, transform=None, is_train=True):
        self.zip_path = zip_path
        self.transform = transform
        self.is_train = is_train

        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            self.image_files = []
            self.labels = []

            image_folder_name = "train_images" if is_train else "test_images"
            for file_info in zip_ref.infolist():
                if file_info.filename.startswith(image_folder_name + "/") and not file_info.filename.endswith("/"):
                    self.image_files.append(file_info)

            if is_train:
                csv_file_name = "labels_train.csv"
                with zip_ref.open(csv_file_name) as f:
                    self.labels_df = pd.read_csv(f)
                    self.labels_dict = dict(zip(self.labels_df['file_name'], self.labels_df['class_id']))
            else:
                self.labels_dict = {}

        self.n_samples = len(self.image_files)

    def __getitem__(self, index):
        max_retries = 5
        retries = 0

        while retries < max_retries:
            image_file_info = self.image_files[index]

            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                with zip_ref.open(image_file_info) as f:
                    try:
                        image = Image.open(f).convert("RGB")
                        if self.transform:
                            image = self.transform(image)
                        image_name = image_file_info.filename.split("/")[-1]
                        label = self.labels_dict.get(image_name)

                        if label is not None:
                            try:
                                label = int(label)
                            except ValueError:
                                pass

                        print(f"Successfully loaded: {image_file_info.filename}")  # Success message
                        return image, label

                    except Exception as e:
                        print(f"Error reading image {image_file_info.filename}: {e}")
                        retries += 1
                        index = (index + 1) % self.n_samples

        print(f"Failed to read image after {max_retries} retries. Skipping.")
        return None

    def __len__(self):
        return self.n_samples


# Image transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

zip_file_path = "C:\\Users\\gulna\\Downloads\\archive.zip"  

# Create datasets and dataloaders
train_dataset = ImageDataset(zip_path=zip_file_path, transform=data_transforms['train'], is_train=True)
test_dataset = ImageDataset(zip_path=zip_file_path, transform=data_transforms['val'], is_train=False)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = None
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)  # Flatten, keeping batch size
        flattened_size = x.size(1)

        if self.fc1 is None:  # Initialize fc1 only once
            self.fc1 = nn.Linear(flattened_size, 120).to(x.device)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
def my_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))  # Filter out None values
    if len(batch) == 0:
        return None, None  # Handle empty batches

    images, labels = zip(*batch)  # Unpack images and labels
    try:
        images = torch.stack(images, 0)
        labels = torch.tensor(labels)
        return images, labels
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        return None, None


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4,drop_last=True,collate_fn=my_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4,drop_last=True,collate_fn=my_collate_fn)

    # Training loop
    n_total_steps = len(train_loader)
    n_epochs = 5
    for epoch in range(n_epochs):
        for i, (images, labels) in enumerate(train_loader):
            if images is None:  # Handle None values
                continue

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{n_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

        # Validation loop
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for images, labels in test_loader:
                if images is None:  # Handle None values
                    continue
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()
            if n_samples>0:  
                acc = 100.0 * n_correct / n_samples
                print(f'Accuracy of the network: {acc} %')
            else:
                print("No valid test samples found.")

    print('Finished Training')

    # Save the model
    with open("cnn.pkl", 'wb') as f:
        pickle.dump(model, f)