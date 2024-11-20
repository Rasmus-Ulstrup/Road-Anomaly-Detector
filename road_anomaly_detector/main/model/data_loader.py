import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from config.config import Config

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
        # Load images and masks
        image = Image.open(self.image_paths[idx]).convert("L")  # Convert to grayscale if required
        mask = Image.open(self.mask_paths[idx]).convert("L")    # Same for masks
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
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
            "image_dir": os.path.expanduser(r"~/Documents/datasetz/datasets/Mixed/Images"),
            "mask_dir": os.path.expanduser(r"~/Documents/datasetz/datasets/Mixed/Masks")
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

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(Config.image_size),
        transforms.ToTensor(),
    ])

    # Create datasets for train, val, and test
    train_dataset = SegmentationDataset(train_images, train_masks, transform=transform)
    val_dataset = SegmentationDataset(val_images, val_masks, transform=transform)
    test_dataset = SegmentationDataset(test_images, test_masks, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader