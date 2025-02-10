import torch
from models.unet_simple import UNet_simple
from models.unet import unet
from models.HED import HED
from models.FPN import FPN
import os
import re

import segmentation_models_pytorch as smp

def sanitize_learning_rate(lr):
    return str(lr).replace('.', '')

class Config:
    def __init__(self, 
                 model_name, 
                 Argumentation=True, 
                 dataset_name=None, 
                 batch_size=4, 
                 image_size=(512,512), 
                 test_size=0.2, 
                 loss_function="dice", 
                 learning_rate=1e-3, 
                 epochs=50, 
                 patience=20,
                 alpha=0.25,
                 gamma=2,
                 preprocessing=False):
        self.model_name = model_name.lower()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.image_size = image_size
        self.test_size = test_size
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Initialize device
        self.model = self.select_model()
        self.argumentation = Argumentation
        self.loss_kwargs = {
            'alpha' : alpha,
            'gamma' : gamma
        }
        self.preprocessing = preprocessing
        

        # Prepare loss suffix based on loss function type
        loss_suffix = self._get_loss_suffix()

        # Sanitize the learning rate
        sanitized_lr = sanitize_learning_rate(self.learning_rate)

        # Construct the base_save_path using the standardized format
        # Format: "{model_name}_{dataset_name}_lr{sanitized_lr}_b{batch_size}_p{patience}_e{epochs}_{loss_suffix}"
        self.base_save_path = os.path.join(
            os.getcwd(), 
            'model_files', 
            f"{self.model_name}_{self.dataset_name}_lr{sanitized_lr}_b{self.batch_size}_p{self.patience}_e{self.epochs}_{loss_suffix}"
        )


        # Assign save paths (modify if different directories are needed)
        self.model_save_path = self.base_save_path
        self.metric_save_path = self.base_save_path
        self.loss_save_path = self.base_save_path

        # Ensure the directory exists (create it if not)
        os.makedirs(os.path.dirname(self.base_save_path), exist_ok=True)

    
    def _get_loss_suffix(self):
        if self.loss_function in ["tversky", "focal"]:
            alpha = self.loss_kwargs.get('alpha', 0.25)
            gamma = self.loss_kwargs.get('gamma', 2)
            return f"{self.loss_function}_alpha{alpha}_gamma{gamma}"
        else:
            return self.loss_function

    def select_model(self):
        if self.model_name == "UNet_simple".lower():
            return UNet_simple().to(self.device)
        elif self.model_name == "UNet".lower():
            return unet().to(self.device)
        elif self.model_name == "HED".lower():
            return HED().to(self.device)
        elif self.model_name == "FPN".lower():
            return FPN().to(self.device)
        elif self.model_name == "smp_unet".lower():  # Example for SMP U-Net
            # # Create the base SMP model
            # base_model = smp.Unet(
            #     encoder_name="resnet34",  # Specify encoder
            #     encoder_weights="imagenet",  # Pre-trained weights
            #     in_channels=1,  # Number of input channels (e.g., grayscale)
            #     classes=1  # Number of output classes
            # )
            # # Add a Sigmoid activation layer to the model
            # return torch.nn.Sequential(base_model, torch.nn.Sigmoid()).to(self.device)
            # Create an unverified SSL context
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            return smp.Unet(
                    encoder_name="densenet201",    # Changed encoder to DenseNet201
                    encoder_weights="imagenet",     # Use pre-trained weights
                    in_channels=1,                  # Grayscale images
                    classes=1,                      # Single output class
                    decoder_channels=(512, 256, 128, 64, 32),  # U-Net decoder channels
                    activation="sigmoid"            # Sigmoid activation for binary segmentation
                ).to(self.device)
        else:
            raise ValueError(f"Model '{self.model_name}' not supported.")
        
    def update_trainer_save_paths(self):
        # make folder if it not exisit:
        os.makedirs(os.path.join(self.base_save_path), exist_ok=True)

        self.model_save_path = os.path.join(self.base_save_path, 'model.pth')
        self.loss_save_path = os.path.join(self.base_save_path, 'loss_plot.png')

    def update_metric_save_path(self, output_dir):
        # make folder if it not exisit:
        os.makedirs(os.path.join(self.base_save_path), exist_ok=True)

        # Edit baseline according to output_dir
        self.metric_save_path = os.path.join(os.path.dirname(output_dir), "metrics.csv")