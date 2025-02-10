import os
from torch.utils.data import Dataset, DataLoader
#from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from config.config import Config
import numpy as np
from metrics.metrics import default_transform
import torch
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2

from metrics.metrics import apply_preprocessing

def calculate_mean_std(image_dir):
    mean = 0.0
    std = 0.0
    count = 0

    for fname in os.listdir(image_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            img_path = os.path.join(image_dir, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue 
            img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
            mean += img.mean()
            std += img.std()
            count += 1

    mean /= count
    std /= count
    return mean, std

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, preprocessing=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask using OpenCV
        image_pre = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        if image_pre is None:
            raise ValueError(f"Image not found or unable to read: {self.image_paths[idx]}")
        

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)  # Load mask as grayscale
        if mask is None:
            raise ValueError(f"Mask not found or unable to read: {self.mask_paths[idx]}")
        # Load images and masks
        #image = Image.open(self.image_paths[idx]).convert("L")  # Convert to grayscale 
        #mask = Image.open(self.mask_paths[idx]).convert("L")

        if self.preprocessing == True:
            image_pre = apply_preprocessing(image_pre)
        
        image_pre = np.expand_dims(image_pre, axis=-1)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image_pre, mask=mask)
            image = augmented['image'].float() / 255
            mask = augmented['mask'].float() 
            mask = (mask > 127).float()  # Binary masks
            
            
            # Display the image and mask
            # plt.figure(figsize=(12, 6))
            # plt.subplot(1, 2, 1)
            # plt.imshow(mask, cmap="gray") #image.numpy().squeeze()
            # plt.title("Transformed Image")
            # plt.axis("off")
            
            # plt.subplot(1, 2, 2)
            # plt.imshow(mask, cmap="gray")
            # plt.title("not transformed Image")
            # plt.axis("off")

            # print("Press any key to close the plot and continue...")
            # plt.show(block=True)
            
            mask = mask.unsqueeze(0)  # Add channel dimension

        return image, mask


def get_data_loaders(Config, preprocessing=False):
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
        "mixed_2": {
            "image_dir": os.path.expanduser(r"~/Documents/datasetz/datasets/Mixed_2/images"),
            "mask_dir": os.path.expanduser(r"~/Documents/datasetz/datasets/Mixed_2/masks")
        },
        "GAPS10m": {
            "image_dir": os.path.expanduser(r"~/Documents/datasetz/datasets/GAPS10m/Images"),
            "mask_dir": os.path.expanduser(r"~/Documents/datasetz/datasets/GAPS10m/Masks")
        }
    }
    
    testsets = {
        "test": {
            "image_dir": os.path.expanduser(r"~/Documents/testsetz/images"),
            "mask_dir": os.path.expanduser(r"~/Documents/testsetz/masks")
        }
    }

     # Validate dataset_name from the config
    if Config.dataset_name not in datasets:
        raise ValueError(f"Dataset '{Config.dataset_name}' not found. Available datasets: {', '.join(datasets.keys())}")

    # Get dataset directories
    image_dir = datasets[Config.dataset_name]["image_dir"]
    mask_dir = datasets[Config.dataset_name]["mask_dir"]

     # Load image and mask paths with compression types
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

      # Split into train (80%), validation (20%)
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )

    # Compute mean and std for normalization based on the training set
    mean, std = calculate_mean_std(image_dir)

    val_transform = default_transform(Config)
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
            A.GaussNoise(std_range=(0.001, 0.005), p=0.5),
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
    
    # Create datasets for train, val, and test (test has been changed to a seperate test folder see report)
    train_dataset = SegmentationDataset(train_images, train_masks, transform=train_transform, preprocessing=preprocessing)
    val_dataset = SegmentationDataset(val_images, val_masks, transform=val_transform, preprocessing=preprocessing)

    test_image_dir = testsets["test"]["image_dir"]
    test_mask_dir = testsets["test"]["mask_dir"]

    test_image_paths = sorted([
        os.path.join(test_image_dir, fname) 
        for fname in os.listdir(test_image_dir) 
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ])
    test_mask_paths = sorted([
        os.path.join(test_mask_dir, fname) 
        for fname in os.listdir(test_mask_dir) 
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ])

    if len(test_image_paths) != len(test_mask_paths):
        raise ValueError("The number of images and masks in testsetz do not match.")
    

    test_dataset = SegmentationDataset(test_image_paths, test_mask_paths, transform=val_transform, preprocessing=preprocessing)

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

    for images, masks in train_loader:
        print(f"Images min: {torch.min(images)} Images max: {torch.max(images)}")
        print(f"Mask min: {torch.min(masks)} Mask max: {torch.max(masks)}")
        break

    return train_loader, val_loader, test_loader