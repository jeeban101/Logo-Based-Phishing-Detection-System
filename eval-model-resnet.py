import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

# Custom dataset class
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

# Image transformations
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the trained model
def load_trained_model(model_path):
    model = models.resnet50(pretrained=False)
    num_classes = 2
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Function to classify a single image
def classify_logo(image_path, model, transform, device):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    classes = ['Fake', 'Genuine'] #adjusted
    return classes[predicted.item()]

# Main execution
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Assuming the model is saved as 'fake_real_logo_detection_model.pth'
    model_path = 'fake_real_logo_detection_model.pth'
    model = load_trained_model(model_path)
    model = model.to(device)

    # Example usage
    # epoch10-resnet/FakeReal Logo Detection dataset/test/Genuine/000001_03da4775bc7540c589813d07d2a60ba7.jpg
    # D:\aaa CAPSTONE\REVIEW-2\efficentNet\epoch10-resnet\FakeReal Logo Detection dataset\test\Genuine\000001_03da4775bc7540c589813d07d2a60ba7.jpg
    # D:\aaa CAPSTONE\REVIEW-2\efficentNet\epoch10-resnet\FakeReal Logo Detection dataset\test\Genuine\000003_4846f680d4d548e997fe6416bf1a950f.jpg
    # epoch10-resnet\FakeReal Logo Detection dataset\test\Genuine\000003_f5d359ca66ee45f89850c3bc985d1e71.jpg
    # epoch10-resnet\FakeReal Logo Detection dataset\test\Fake\000001_4af57d3130914c77855af48f705c9396.jpg
    test_image_path = 'FakeReal Logo Detection dataset/test/Genuine/000001_03da4775bc7540c589813d07d2a60ba7.jpg'  # Update this path
    predicted_class = classify_logo(test_image_path, model, image_transforms, device)
    print("Predicted class:", predicted_class)

    

