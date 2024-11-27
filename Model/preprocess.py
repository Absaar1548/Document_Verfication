from PIL import Image
import numpy as np
import torch
from torchvision import transforms

def preprocess_image(image_path: str, resize: int | list = 128):
    """
    Preprocesses an image for the model input.

    Args:
        image_path (str): Path to the image file.
        resize (int): The target size to resize the image.

    Returns:
        torch.Tensor: The preprocessed image tensor.
    """
    # Load image
    image = Image.open(image_path).convert("L")

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for pre-trained models
    ])

    # Apply transformations
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

def preprocess_images(image_path1: str, image_path2: str, resize: int):
    """
    Preprocesses two images for signature matching.

    Args:
        image_path1 (str): Path to the first image.
        image_path2 (str): Path to the second image.
        resize (int): The target size to resize the images.

    Returns:
        torch.Tensor: Two preprocessed image tensors.
    """
    img1 = preprocess_image(image_path1, resize)
    img2 = preprocess_image(image_path2, resize)
    return img1, img2
