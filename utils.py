import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import __main__  # we’ll use this to “register” BrainTumorNet under __main__

# --------------------------
# 1. Define your custom model class
# --------------------------
class BrainTumorNet(nn.Module):
    def __init__(self, num_classes):
        super(BrainTumorNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=32 * 56 * 56, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, input_tensor):
        x = self.conv_layers(input_tensor)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# --------------------------
# 2. Define label mapping
# --------------------------
LABELS = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}

# --------------------------
# 3. Define transform pipeline
# --------------------------
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --------------------------
# 4. Full model loader
# --------------------------
def load_model(model_path: str):
    """
    Load the entire saved model (architecture + weights). 
    We must register BrainTumorNet under __main__ so that torch.load can unpickle it.
    """
    # 1) “Alias” BrainTumorNet into __main__ so that pickle.find_class("__main__", "BrainTumorNet") works:
    __main__.BrainTumorNet = BrainTumorNet

    # 2) Now load the model (saved via torch.save(model))
    model = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))
    model.eval()
    return model

# --------------------------
# 5. Prediction function
# --------------------------
def predict(model, image: Image.Image):
    """
    Preprocess the PIL image and run inference.
    Returns the predicted label string.
    """
    img_tensor = transform_pipeline(image).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    return LABELS[predicted.item()]
