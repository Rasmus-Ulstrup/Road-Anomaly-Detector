import torch
from models.unet_simple import UNet_simple
from models.unet import unet
from models.HED import HED
from models.FPN import FPN
import os
import re

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
                 gamma=2):
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
        # Define the model save path relative to the current working directory
        self.base_save_path = os.path.join(os.getcwd(), 'model_files', 
                                            f"{model_name}_{dataset_name}_lr{str(learning_rate).replace('.', '')}_b{batch_size}_p{patience}_e{epochs}_{loss_function}")
        self.model_save_path = "NOT CHANGED"
        self.metric_save_path = "NOT CHANGED"
        self.loss_save_path = "NOT CHANGED"
        # Ensure the directory exists (create it if not)
        os.makedirs(os.path.join(os.getcwd(), 'model_files'), exist_ok=True)

    def select_model(self):
        if self.model_name == "UNet_simple".lower():
            return UNet_simple().to(self.device)
        elif self.model_name == "UNet".lower():
            return unet().to(self.device)
        elif self.model_name == "HED".lower():
            return HED().to(self.device)
        elif self.model_name == "FPN".lower():
            return FPN().to(self.device)
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