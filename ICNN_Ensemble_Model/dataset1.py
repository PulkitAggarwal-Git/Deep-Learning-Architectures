import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from data import get_data

training_data, validation_data, testing_data = get_data()

class_names = ['1','2','3']

transformations_cnn1 = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
])

class Dataset_CNN1(Dataset):
    def __init__(self, data, transform=None, crop_scale=0.85):
        self.data = data
        self.transform = transform
        self.crop_scale = crop_scale

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, _, label = self.data[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            transformed_image = self.transform(image)
        else:
            transformed_image = image

        label_idx = class_names.index(label)

        return transformed_image, label_idx
    
train_dataset_cnn1 = Dataset_CNN1(training_data, transform=transformations_cnn1)
val_dataset_cnn1 = Dataset_CNN1(validation_data, transform=transformations_cnn1)
test_dataset_cnn1 = Dataset_CNN1(testing_data, transform=transformations_cnn1)

train_loader_cnn1 = DataLoader(train_dataset_cnn1, batch_size=32, shuffle=True)
val_loader_cnn1 = DataLoader(val_dataset_cnn1, batch_size=32, shuffle=False)
test_loader_cnn1 = DataLoader(test_dataset_cnn1, batch_size=32, shuffle=False)