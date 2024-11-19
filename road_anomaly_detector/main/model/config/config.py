import torch
from models.unet_simple import UNet_simple
from models.unet_advanced import UNet_advanced
import os
class Config:
    def __init__(self, model_name, dataset_name=None, batch_size=4, image_size=(448,448), test_size=0.2, loss_function="dice", learning_rate=1e-3, epochs=50, patience=20):
        self.model_name = model_name
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
        self.model_save_path = os.path.join('model_files', f"{model_name}_{dataset_name}_lr{str(learning_rate).replace('.', '')}_b{batch_size}_p{patience}_e{epochs}_{loss_function}.pth")


    def select_model(self):
        if self.model_name == "UNet_simple":
            return UNet_simple().to(self.device)
        elif self.model_name == "UNet_advanced":
            return UNet_advanced().to(self.device)
        else:
            raise ValueError(f"Model '{self.model_name}' not supported.")