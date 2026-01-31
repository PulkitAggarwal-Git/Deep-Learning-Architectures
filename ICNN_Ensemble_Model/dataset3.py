import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from data import get_data

training_data, validation_data, testing_data = get_data()

class_names = ['1','2','3']

transformations_cnn3 = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[1:2, :, :]),
    ])

class Dataset_CNN3(Dataset):
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
    
train_dataset_cnn3 = Dataset_CNN3(training_data, transform=transformations_cnn3)
val_dataset_cnn3 = Dataset_CNN3(validation_data, transform=transformations_cnn3)
test_dataset_cnn3 = Dataset_CNN3(testing_data, transform=transformations_cnn3)

train_loader_cnn3 = DataLoader(train_dataset_cnn3, batch_size=32, shuffle=True)
val_loader_cnn3 = DataLoader(val_dataset_cnn3, batch_size=32, shuffle=False)
test_loader_cnn3 = DataLoader(test_dataset_cnn3, batch_size=32, shuffle=False)