import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import models, datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image, ImageDraw
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

num_classes = len(os.listdir('/kaggle/input/liver-diseasecirrhosis-dataset/Liver Disease Dataset/train'))
class_names = os.listdir('/kaggle/input/liver-diseasecirrhosis-dataset/Liver Disease Dataset/train')

def extract_data(root_dir):
    data = []

    for label in os.listdir(root_dir):
        label_dir = os.path.join(root_dir,label)
        for patient in os.listdir(label_dir):
            patient_dir = os.path.join(label_dir, patient)
            images_dir = os.path.join(patient_dir, 'images')
    
            for filename in os.listdir(images_dir):
                image_path = os.path.join(images_dir, filename)
                data.append([image_path, patient, label])
    
    return data

train_dir = '/kaggle/input/liver-diseasecirrhosis-dataset/Liver Disease Dataset/train'
test_dir = '/kaggle/input/liver-diseasecirrhosis-dataset/Liver Disease Dataset/test'
validation_dir = '/kaggle/input/liver-diseasecirrhosis-dataset/Liver Disease Dataset/validation'

training_data = extract_data(train_dir)
validation_data = extract_data(validation_dir)
testing_data = extract_data(test_dir)

def elliptical_crop(image, scale=0.85):
    width, height = image.size
    ellipse_width = width * scale
    ellipse_height = height * scale
    center_x, center_y = width // 2, height // 2

    # Create an elliptical mask
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(
        (
            center_x - ellipse_width // 2,
            center_y - ellipse_height // 2,
            center_x + ellipse_width // 2,
            center_y + ellipse_height // 2,
        ),
        fill=255,
    )

    # Apply the mask
    result = Image.new("RGBA", image.size)
    result.paste(image, (0, 0), mask)

    # Crop the bounding box of the ellipse
    cropped_image = result.crop((
        center_x - ellipse_width // 2,
        center_y - ellipse_height // 2,
        center_x + ellipse_width // 2,
        center_y + ellipse_height // 2,
    ))

    return cropped_image.convert("RGB")

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

        cropped_image = elliptical_crop(image, scale=self.crop_scale)

        if self.transform:
            transformed_image = self.transform(cropped_image)
        else:
            transformed_image = cropped_image

        label_idx = class_names.index(label)

        return transformed_image, label_idx
    
train_dataset_cnn1 = Dataset_CNN1(training_data, transform=transformations_cnn1, crop_scale=0.85)
val_dataset_cnn1 = Dataset_CNN1(validation_data, transform=transformations_cnn1, crop_scale=0.85)
test_dataset_cnn1 = Dataset_CNN1(testing_data, transform=transformations_cnn1, crop_scale=0.85)

train_loader_cnn1 = DataLoader(train_dataset_cnn1, batch_size=32, shuffle=True)
val_loader_cnn1 = DataLoader(val_dataset_cnn1, batch_size=32, shuffle=False)
test_loader_cnn1 = DataLoader(test_dataset_cnn1, batch_size=32, shuffle=False)

transformations_cnn2 = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[0:1, :, :]),
    ])

class Dataset_CNN2(Dataset):
    def __init__(self, data, transform=None, crop_scale=0.85):
        self.data = data
        self.transform = transform
        self.crop_scale = crop_scale

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, _, label = self.data[idx]
        image = Image.open(image_path).convert('RGB')

        cropped_image = elliptical_crop(image, scale=self.crop_scale)

        if self.transform:
            transformed_image = self.transform(cropped_image)
        else:
            transformed_image = cropped_image

        label_idx = class_names.index(label)

        return transformed_image, label_idx
    
train_dataset_cnn2 = Dataset_CNN2(training_data, transform=transformations_cnn2, crop_scale=0.85)
val_dataset_cnn2 = Dataset_CNN2(validation_data, transform=transformations_cnn2, crop_scale=0.85)
test_dataset_cnn2 = Dataset_CNN2(testing_data, transform=transformations_cnn2, crop_scale=0.85)

train_loader_cnn2 = DataLoader(train_dataset_cnn2, batch_size=32, shuffle=True)
val_loader_cnn2 = DataLoader(val_dataset_cnn2, batch_size=32, shuffle=False)
test_loader_cnn2 = DataLoader(test_dataset_cnn2, batch_size=32, shuffle=False)

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

        cropped_image = elliptical_crop(image, scale=self.crop_scale)

        if self.transform:
            transformed_image = self.transform(cropped_image)
        else:
            transformed_image = cropped_image

        label_idx = class_names.index(label)

        return transformed_image, label_idx
    
train_dataset_cnn3 = Dataset_CNN3(training_data, transform=transformations_cnn3, crop_scale=0.85)
val_dataset_cnn3 = Dataset_CNN3(validation_data, transform=transformations_cnn3, crop_scale=0.85)
test_dataset_cnn3 = Dataset_CNN3(testing_data, transform=transformations_cnn3, crop_scale=0.85)

train_loader_cnn3 = DataLoader(train_dataset_cnn3, batch_size=32, shuffle=True)
val_loader_cnn3 = DataLoader(val_dataset_cnn3, batch_size=32, shuffle=False)
test_loader_cnn3 = DataLoader(test_dataset_cnn3, batch_size=32, shuffle=False)

transformations_cnn4 = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[2:3, :, :]),
    ])

class Dataset_CNN4(Dataset):
    def __init__(self, data, transform=None, crop_scale=0.85):
        self.data = data
        self.transform = transform
        self.crop_scale = crop_scale

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, _, label = self.data[idx]
        image = Image.open(image_path).convert('RGB')

        cropped_image = elliptical_crop(image, scale=self.crop_scale)

        if self.transform:
            transformed_image = self.transform(cropped_image)
        else:
            transformed_image = cropped_image

        label_idx = class_names.index(label)

        return transformed_image, label_idx
    
train_dataset_cnn4 = Dataset_CNN4(training_data, transform=transformations_cnn4, crop_scale=0.85)
val_dataset_cnn4 = Dataset_CNN4(validation_data, transform=transformations_cnn4, crop_scale=0.85)
test_dataset_cnn4 = Dataset_CNN4(testing_data, transform=transformations_cnn4, crop_scale=0.85)

train_loader_cnn4 = DataLoader(train_dataset_cnn4, batch_size=32, shuffle=True)
val_loader_cnn4 = DataLoader(val_dataset_cnn4, batch_size=32, shuffle=False)
test_loader_cnn4 = DataLoader(test_dataset_cnn4, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512, 256)
        self.linear1 = nn.Linear(256,128)
        self.linear2 = nn.Linear(128,3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)
        x = self.maxpool(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
       
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        
        x = self.flatten(x)
        x = F.relu(self.linear(x))
        
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        
        return x
    
cnn1 = Model(3)
cnn2 = Model(1)
cnn3 = Model(1)
cnn4 = Model(1)

def train_cnn1(model, train_loader_cnn1, val_loader_cnn1, epochs=50, device=device):
    print("-------------CNN-1 Training---------------")
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        all_preds, all_labels = [], []

        for inputs, labels in train_loader_cnn1:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        train_loss /= len(train_loader_cnn1)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        train_acc = (all_preds == all_labels).float().mean().item()
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Acc: {train_acc:.4f} | Loss: {train_loss:.4f}")

        # ----------------- VALIDATION (every 10 epochs) ----------------
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for inputs, labels in val_loader_cnn1:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    logits = model(inputs)
                    loss = criterion(logits, labels)

                    val_loss += loss.item()
                    preds = logits.argmax(dim=1)

                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())

            val_loss /= len(val_loader_cnn1)

            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()

            val_acc = (all_preds == all_labels).mean()

            print(f"Epoch [{epoch+1}/{epochs}] | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

train_cnn1(cnn1, train_loader_cnn1, val_loader_cnn1, device=device)

def train_cnn2(model, train_loader_cnn2, val_loader_cnn2, epochs=50, device=device):
    print("-------------CNN-2 Training---------------")
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        all_preds, all_labels = [], []

        for inputs, labels in train_loader_cnn2:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        train_loss /= len(train_loader_cnn2)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        train_acc = (all_preds == all_labels).float().mean().item()
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Acc: {train_acc:.4f} | Loss: {train_loss:.4f}")

        # ---------------- VALIDATION (every 10 epochs) ------------------
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for inputs, labels in val_loader_cnn2:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    logits = model(inputs)
                    loss = criterion(logits, labels)

                    val_loss += loss.item()
                    preds = logits.argmax(dim=1)

                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())

            val_loss /= len(val_loader_cnn2)

            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()

            val_acc = (all_preds == all_labels).mean()

            print(f"Epoch [{epoch+1}/{epochs}] | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

train_cnn2(cnn2, train_loader_cnn2, val_loader_cnn2, device=device)

def train_cnn3(model, train_loader_cnn3, val_loader_cnn3, epochs=50, device=device):
    print("-------------CNN-3 Training---------------")
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        all_preds, all_labels = [], []

        for inputs, labels in train_loader_cnn3:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        train_loss /= len(train_loader_cnn3)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        train_acc = (all_preds == all_labels).float().mean().item()

        print(f"Epoch [{epoch+1}/{epochs}] | Train Acc: {train_acc:.4f} | Loss: {train_loss:.4f}")

        # ----------------- VALIDATION (every 10 epochs) -----------------
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for inputs, labels in val_loader_cnn3:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    logits = model(inputs)
                    loss = criterion(logits, labels)

                    val_loss += loss.item()
                    preds = logits.argmax(dim=1)

                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())

            val_loss /= len(val_loader_cnn3)

            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()

            val_acc = (all_preds == all_labels).mean()

            print(f"Epoch [{epoch+1}/{epochs}] | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

train_cnn3(cnn3, train_loader_cnn3, val_loader_cnn3, device=device)

def train_cnn4(model, train_loader_cnn4, val_loader_cnn4, epochs=50, device=device):
    print("-------------CNN-4 Training---------------")
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        all_preds, all_labels = [], []

        for inputs, labels in train_loader_cnn4:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        train_loss /= len(train_loader_cnn4)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        train_acc = (all_preds == all_labels).float().mean().item()

        print(f"Epoch [{epoch+1}/{epochs}] | Train Acc: {train_acc:.4f} | Loss: {train_loss:.4f}")

        # ---------------- VALIDATION (every 10 epochs) ----------------
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for inputs, labels in val_loader_cnn4:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    logits = model(inputs)
                    loss = criterion(logits, labels)

                    val_loss += loss.item()
                    preds = logits.argmax(dim=1)

                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())

            val_loss /= len(val_loader_cnn4)

            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()

            val_acc = (all_preds == all_labels).mean()

            print(f"Epoch [{epoch+1}/{epochs}] | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

train_cnn4(cnn4, train_loader_cnn4, val_loader_cnn4, device=device)

def test_cnn1(model, test_loader_cnn1, device=device):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader_cnn1:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            test_loss += loss.item()
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    test_loss /= len(test_loader_cnn1)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    test_acc = (all_preds == all_labels).float().mean().item()

    return test_acc

def test_cnn2(model, test_loader_cnn2, device=device):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader_cnn2:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            test_loss += loss.item()
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    test_loss /= len(test_loader_cnn2)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    test_acc = (all_preds == all_labels).float().mean().item()

    return test_acc

def test_cnn3(model, test_loader_cnn3, device=device):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader_cnn3:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            test_loss += loss.item()
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    test_loss /= len(test_loader_cnn3)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    test_acc = (all_preds == all_labels).float().mean().item()

    return test_acc

def test_cnn4(model, test_loader_cnn4, device=device):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader_cnn4:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            test_loss += loss.item()
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    test_loss /= len(test_loader_cnn4)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    test_acc = (all_preds == all_labels).float().mean().item()

    return test_acc

acc1 = test_cnn1(cnn1, test_loader_cnn1, device=device)
acc2 = test_cnn2(cnn2, test_loader_cnn2, device=device)
acc3 = test_cnn2(cnn3, test_loader_cnn3, device=device)
acc4 = test_cnn2(cnn4, test_loader_cnn4, device=device)

models_acc = {
    "CNN1" : acc1,
    "CNN2" : acc2,
    "CNN3" : acc3,
    "CNN4" : acc4
}

def smde_select(models_acc, main_model_name="CNN1", b=True):
    best_acc = max(models_acc.values())
    threshold = 0.04 if b else 0.03

    selected_models = []

    for model_name, acc in models_acc.items():
        if model_name == main_model_name:
            selected_models.append(model_name)
        elif acc == best_acc:
            selected_models.append(model_name)
        elif acc >= best_acc - threshold:
            selected_models.append(model_name)

    return selected_models

selected_models = smde_select(models_acc)
print("Selected models after SMDE:", selected_models)

model_info = {
    "CNN1": {"model": cnn1, "loader": test_loader_cnn1},
    "CNN2": {"model": cnn2, "loader": test_loader_cnn2},
    "CNN3": {"model": cnn3, "loader": test_loader_cnn3},
    "CNN4": {"model": cnn4, "loader": test_loader_cnn4},
}

def ensemble_inference(model_info, selected_models, device):
    all_probs = []
    all_labels = None 

    for name in selected_models:
        model = model_info[name]["model"].to(device)
        loader = model_info[name]["loader"]

        model.eval()
        probs_list = []
        labels_list = []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)

                logits = model(inputs)              # (B, 3)
                probs = F.softmax(logits, dim=1)    # (B, 3)

                probs_list.append(probs.cpu())
                labels_list.append(labels.cpu())

        probs_list = torch.cat(probs_list, dim=0)   # (N, 3)
        all_probs.append(probs_list)

        # collect labels only once (same order for all models)
        if all_labels is None:
            all_labels = torch.cat(labels_list, dim=0)

    # Ensemble probability aggregation 
    ensemble_probs = torch.stack(all_probs, dim=0).sum(dim=0)  # (N, 3)
    preds = ensemble_probs.argmax(dim=1)                       # (N,)

    # Metrics 
    y_true = all_labels.numpy()
    y_pred = preds.numpy()

    cm = confusion_matrix(y_true, y_pred)
    accuracy = (y_true == y_pred).mean()

    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)  # sensitivity
    f1        = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Specificity (multi-class)
    specificity_per_class = []
    for i in range(cm.shape[0]):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        specificity_per_class.append(TN / (TN + FP + 1e-8))

    specificity = np.mean(specificity_per_class)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,         
        "specificity": specificity,
        "f1": f1,
        "confusion_matrix": cm
    }

results = ensemble_inference(model_info, selected_models, device)

print("Accuracy     :", results["accuracy"])
print("Precision    :", results["precision"])
print("Recall       :", results["recall"])
print("Specificity  :", results["specificity"])
print("F1-score     :", results["f1"])
print("Confusion Matrix:\n", results["confusion_matrix"])