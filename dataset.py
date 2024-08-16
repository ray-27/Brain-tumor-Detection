import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomImageDataset(Dataset):
    """
    A custom dataset class that loads images from a directory, applying transformations
    and returning images with their corresponding labels.
    """
    def __init__(self, image_dir, transform=None):
        """
        Initializes the dataset object.
        :param image_dir: Path to the directory containing images.
        :param transform: Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        :param idx: Index of the sample to retrieve.
        """
        img_name = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

# Usage example
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CustomImageDataset(image_dir='path/to/your/image/directory', transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

    # Iterate through the data loader
    for images in data_loader:
        # Perform operations using images
        print(images.shape)  # Output the shape of the images tensor
