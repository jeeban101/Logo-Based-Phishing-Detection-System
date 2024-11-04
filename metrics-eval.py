import torch
from torchvision import transforms, models
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image


#from custom_logo_dataset import CustomLogoDataset  # Assuming this is saved in a separate file named custom_logo_dataset.py

class CustomLogoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for label, subdir in enumerate(['Fake', 'Genuine']):
            folder_path = os.path.join(self.root_dir, subdir)
            for filename in os.listdir(folder_path):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    self.images.append(os.path.join(folder_path, filename))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def load_trained_model(model_path):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)  # Assuming 2 classes: Fake and Genuine
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_dataset = CustomLogoDataset('FakeReal Logo Detection dataset/test', transform=image_transforms)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def evaluate_model(model, dataloader, device):
    y_pred = []
    y_true = []

    model = model.to(device)
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.view(-1).tolist())
            y_true.extend(labels.view(-1).tolist())

    print(classification_report(y_true, y_pred, target_names=['Fake', 'Genuine']))

    return y_true, y_pred


def plot_confusion_matrix(y_true, y_pred):
    cf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Genuine'], yticklabels=['Fake', 'Genuine'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'fake_real_logo_detection_model.pth'
    model = load_trained_model(model_path)
    
    y_true, y_pred = evaluate_model(model, test_dataloader, device)
    plot_confusion_matrix(y_true, y_pred)

