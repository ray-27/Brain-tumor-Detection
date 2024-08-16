import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load your trained model
model = SimpleCNN(num_classes=10)  # Adjust num_classes based on your model
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load an image
image_path = 'path/to/your/image.jpg'
image = Image.open(image_path).convert('RGB')
image = transform(image)
image = image.unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)

# Display the image and prediction
plt.imshow(Image.open(image_path))
plt.title(f'Predicted Class: {predicted.item()}')
plt.show()