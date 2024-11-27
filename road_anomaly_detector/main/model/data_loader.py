import os
from torch.utils.data import Dataset, DataLoader
#from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from config.config import Config
import numpy as np

import torch
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2

def calculate_mean_std(image_dir):
    """
    Calculate the mean and standard deviation of greyscale images in a directory.

    Args:
        image_dir (str): Path to the directory containing images.

    Returns:
        mean (float): Mean of the dataset.
        std (float): Standard deviation of the dataset.
    """
    mean = 0.0
    std = 0.0
    count = 0

    for fname in os.listdir(image_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            img_path = os.path.join(image_dir, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue  # Skip corrupted images
            img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
            mean += img.mean()
            std += img.std()
            count += 1

    mean /= count
    std /= count
    return mean, std

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        """
        Args:
            image_paths (list): List of image file paths.
            mask_paths (list): List of mask file paths.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask using OpenCV
        image_pre = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)  # Loads as BGR by default
        if image_pre is None:
            raise ValueError(f"Image not found or unable to read: {self.image_paths[idx]}")
        image_pre = np.expand_dims(image_pre, axis=-1)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)  # Load mask as grayscale
        if mask is None:
            raise ValueError(f"Mask not found or unable to read: {self.mask_paths[idx]}")
        # Load images and masks
        #image = Image.open(self.image_paths[idx]).convert("L")  # Convert to grayscale if required
        #mask = Image.open(self.mask_paths[idx]).convert("L")    # Same for masks
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image_pre, mask=mask)
            image = augmented['image'].float() / 255
            mask = augmented['mask'].float()  # Ensure mask is float32 can be normalized with / 255.0

            # # Display the image and mask
            # plt.figure(figsize=(12, 6))
            # plt.subplot(1, 2, 1)
            # plt.imshow(image.numpy().squeeze(), cmap="gray")
            # plt.title("Transformed Image")
            # plt.axis("off")

            # plt.subplot(1, 2, 2)
            # plt.imshow(image_pre, cmap="gray")
            # plt.title("not transformed Image")
            # plt.axis("off")

            # print("Press any key to close the plot and continue...")
            # plt.show(block=True)
            
            mask = (mask > 0.5).float()  # Binary masks
            mask = mask.unsqueeze(0)  # Add channel dimension

        return image, mask


def get_data_loaders(Config):
    """
    Args:
        dataset_name (str): The name of the dataset to load (e.g., 'cracktree200', 'forest', 'gaps384', 'cfd', 'mixed').
        batch_size (int): Batch size for the data loaders.
        image_size (tuple): Desired image size for resizing (default is 448x320).
        test_size (float): Proportion of the data to be used for validation (default is 0.2).
        random_state (int): Seed for reproducibility during data splitting.

    Returns:
        train_loader, val_loader: DataLoaders for training and validation.
    """
    # Define the dataset directories
    datasets = {
        "cracktree200": {
            "image_dir": os.path.expanduser(r"~/Documents/datasetz/datasets/cracktree200/Images"),
            "mask_dir": os.path.expanduser(r"~/Documents/datasetz/datasets/cracktree200/Masks")
        },
        "forest": {
            "image_dir": os.path.expanduser(r"~/Documents/datasetz/datasets/forest/Images"),
            "mask_dir": os.path.expanduser(r"~/Documents/datasetz/datasets/forest/Masks")
        },
        "gaps384": {
            "image_dir": os.path.expanduser(r"~/Documents/datasetz/datasets/GAPS384/Images"),
            "mask_dir": os.path.expanduser(r"~/Documents/datasetz/datasets/GAPS384/Masks")
        },
        "cfd": {
            "image_dir": os.path.expanduser(r"~/Documents/datasetz/datasets/CrackForestDataset_(CFD)/Images"),
            "mask_dir": os.path.expanduser(r"~/Documents/datasetz/datasets/CrackForestDataset_(CFD)/Masks")
        },
        "mixed": {
            "image_dir": os.path.expanduser(r"~/Documents/datasetz/datasets/Mixed/images"),
            "mask_dir": os.path.expanduser(r"~/Documents/datasetz/datasets/Mixed/masks")
        },
        "GAPS10m": {
            "image_dir": os.path.expanduser(r"~/Documents/datasetz/datasets/GAPS10m/Images"),
            "mask_dir": os.path.expanduser(r"~/Documents/datasetz/datasets/GAPS10m/Masks")
        }
    }
    
    testsets = {
        "cracktree200": {
            "image_dir": os.path.expanduser(r"~/Documents/datasetz/datasets/Test_sets/cracktree200/Images"),
            "mask_dir": os.path.expanduser(r"~/Documents/datasetz/datasets/Test_sets/cracktree200/Masks")
        },
        "forest": {
            "image_dir": os.path.expanduser(r"~/Documents/datasetz/datasets/Test_sets/forest/Images"),
            "mask_dir": os.path.expanduser(r"~/Documents/datasetz/datasets/Test_sets/forest/Masks")
        },
        "gaps384": {
            "image_dir": os.path.expanduser(r"~/Documents/datasetz/datasets/Test_sets/GAPS384/Images"),
            "mask_dir": os.path.expanduser(r"~/Documents/datasetz/datasets/Test_sets/GAPS384/Masks")
        },
        "cfd": {
            "image_dir": os.path.expanduser(r"~/Documents/datasetz/datasets/Test_sets/CrackForestDataset_(CFD)/Images"),
            "mask_dir": os.path.expanduser(r"~/Documents/datasetz/datasets/Test_sets/CrackForestDataset_(CFD)/Masks")
        },
        "mixed": {
            "image_dir": os.path.expanduser(r"~/Documents/datasetz/datasets/Test_sets/Mixed/Images"),
            "mask_dir": os.path.expanduser(r"~/Documents/datasetz/datasets/Test_sets/Mixed/Masks")
        }
    }

     # Validate dataset_name from the config
    if Config.dataset_name not in datasets:
        raise ValueError(f"Dataset '{Config.dataset_name}' not found. Available datasets: {', '.join(datasets.keys())}")

    # Get dataset directories
    image_dir = datasets[Config.dataset_name]["image_dir"]
    mask_dir = datasets[Config.dataset_name]["mask_dir"]

     # Load image and mask paths with common image extensions
    image_paths = sorted([
        os.path.join(image_dir, fname) 
        for fname in os.listdir(image_dir) 
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ])
    mask_paths = sorted([
        os.path.join(mask_dir, fname) 
        for fname in os.listdir(mask_dir) 
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ])

    # Ensure that the number of images and masks match
    if len(image_paths) != len(mask_paths):
        raise ValueError("The number of images and masks do not match.")

    # Load image and mask paths
    image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
    mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)])

    # Split into train, validation, and test sets
    train_images, temp_images, train_masks, temp_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    val_images, test_images, val_masks, test_masks = train_test_split(
        temp_images, temp_masks, test_size=0.5, random_state=42
    )

    # Compute mean and std for normalization based on the training set
    train_image_dir = image_dir  # Ensure this points to the actual training images
    mean, std = calculate_mean_std(train_image_dir)

    # Define transformations
    val_transform = A.Compose([
        A.Resize(height=Config.image_size[0], width=Config.image_size[1]),
        #A.Normalize(mean=(mean,), std=(std,)),
        ToTensorV2()
    ])
    if Config.argumentation == 1:
        print("Training with argumentation")
        train_transform = A.Compose([
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=(0.25, 0.75),
                rotate_limit=30,
                p=0.75
            ),
            A.GaussNoise(var_limit=(0.001, 0.005), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.15, 0.15),
                contrast_limit=(-0.15, 0.15),
                p=0.75
            ),
            A.RandomGamma(gamma_limit=(80, 100), p=0.75),
            #A.Blur(blur_limit=3, p=0.75),
            #A.MotionBlur(blur_limit=3, p=0.75),
            A.Resize(height=Config.image_size[0], width=Config.image_size[1]),
            #A.Normalize(mean=(mean,), std=(std,)),
            ToTensorV2()
        ])
    else:
        print("Training without argumentation")
        train_transform = val_transform
    
    # Create datasets for train, val, and test
    train_dataset = SegmentationDataset(train_images, train_masks, transform=train_transform)
    val_dataset = SegmentationDataset(val_images, val_masks, transform=val_transform)
    test_dataset = SegmentationDataset(test_images, test_masks, transform=val_transform)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    # for images, masks in train_loader:
    #     print(f"Images dtype: {images.dtype}")  # Should be torch.float32
    #     print(f"Masks dtype: {masks.dtype}")    # Should be torch.float32
    #     break

    # for images, masks in train_loader:
    #     print(f"Images shape: {images.shape}")  # Expected: [batch_size, 3, H, W]
    #     print(f"Masks shape: {masks.shape}")    # Expected: [batch_size, 1, H, W]
    #     break
    for images, masks in train_loader:
        print(f"Images min: {torch.min(images)} Images max: {torch.max(images)}")  # Expected: [batch_size, 3, H, W]
        print(f"Mask min: {torch.min(masks)} Mask max: {torch.max(masks)}")  # Expected: [batch_size, 3, H, W]
        break

    return train_loader, val_loader, test_loader