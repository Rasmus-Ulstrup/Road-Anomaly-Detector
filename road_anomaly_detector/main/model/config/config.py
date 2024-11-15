import torch
from models.unet_simple import UNet_simple
from models.unet_advanced import UNet_advanced

class Config:
    def __init__(self, model_name='UNet_simple'):
        # Device (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Dynamically set the model based on the parameter
        self.model_name = model_name
        self.model = self.select_model()
    
    def select_model(self):
        """Select the model based on the provided model_name"""
        if self.model_name == 'UNet_simple':
            return UNet_simple().to(self.device)
        elif self.model_name == 'UNet_advanced':
            return UNet_advanced().to(self.device)
        else:
            raise ValueError(f"Model '{self.model_name}' not supported")
