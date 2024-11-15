import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms, models
from PIL import Image
import os
from torchvision.models import EfficientNet_B4_Weights

# Model configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.abspath('controllers/face_shape/best_model.pth')
NUM_CLASSES = 5  # Update with your number of classes
CLASS_LABELS = ['Oval', 'Square', 'Round', 'Heart', 'Rectangle']  # Update as per your dataset

# Load the model once when the module is imported
model = torchvision.models.efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
  # Initialize EfficientNet without pretrained weights
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.3, inplace=True),
    torch.nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_face_shape(file):
    """
    Predict the face shape for an uploaded image file.
    
    Args:
        file (werkzeug.datastructures.FileStorage): Uploaded image file.
    
    Returns:
        dict: Predicted class, confidence, and probabilities for all classes.
    """
    # Load and preprocess the image
    image = Image.open(file).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Perform inference
    with torch.inference_mode():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    # Convert results to a dictionary
    result = {
        'predicted_class': CLASS_LABELS[predicted_class.item()],
        'confidence': f"{confidence.item() * 100:.2f}%",
        'class_probabilities': {
            CLASS_LABELS[i]: f"{prob.item() * 100:.2f}%" for i, prob in enumerate(probabilities.cpu().numpy().flatten())
        }
    }
    return result
