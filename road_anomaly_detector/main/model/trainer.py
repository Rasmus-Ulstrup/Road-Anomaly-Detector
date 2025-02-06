import torch
import torch.nn.functional as F
from config.config import Config
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pprint


class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.config = config 
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.device 
        self.num_epochs = config.epochs
        self.learning_rate = config.learning_rate
        self.patience = config.patience
        self.model_save_path = config.model_save_path
        self.loss_save_path = config.loss_save_path

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Select loss function
        self.loss_function = self.select_loss_function(config.loss_function, **config.loss_kwargs)

        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        # Initialize lists to store losses
        self.train_losses = []
        self.val_losses = []

        # Print all variables
        self.print_trainer_variables()

    def print_trainer_variables(self):
        """Print all variables of the Trainer class in a structured format."""
        variables = {
            "Device": self.device,
            "Number of Epochs": self.num_epochs,
            "Learning Rate": self.learning_rate,
            "Patience": self.patience,
            "Model Save Path": self.model_save_path,
            "Loss Save Path": self.loss_save_path,
            "Optimizer": str(self.optimizer),
            "Loss Function": self.loss_function.__name__ if hasattr(self.loss_function, '__name__') else str(self.loss_function),
            "Best Validation Loss": self.best_val_loss,
        }
        print("\n" + "=" * 50)
        print("Trainer Configuration")
        print("=" * 50)
        pprint.pprint(variables, indent=4, width=80)
        print("=" * 50 + "\n")

    def select_loss_function(self, loss_type, **kwargs):
        """Select the loss function based on the string type."""
        if loss_type == "dice":
            return self.dice_loss
        elif loss_type == "bce":
            # Set alpha to 0.9 by default, adjust based on dataset
            pos_weight = kwargs.get('pos_weight', 0.9)
            pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(self.device)
            return torch.nn.BCELoss()
        elif loss_type == "ce":
            return self.cross_entropy_loss()
        elif loss_type == "focal":
            alpha = kwargs.get('alpha', 0.25)  # Alpha to mittigate class imbalance
            gamma = kwargs.get('gamma', 2)     #Gamma for focusing on hard examples
            print(f"Using Focal with alpha: {alpha} and gamma: {gamma}")
            return self.focal_loss(gamma)
        elif loss_type == "tversky":
            alpha = kwargs.get('alpha', 0.3)
            beta = kwargs.get('gamma', 0.7) 
            print(f"Using tversky with alpha: {alpha} and beta: {beta}")
            return self.tversky_loss(alpha, beta)
        else:
            raise ValueError(f"Loss function {loss_type} not supported")

    def dice_loss(self, pred, target, smooth=1e-5): #default smoothing
        pred = pred.contiguous()
        target = target.contiguous()
        
        # Compute intersection and sums over spatial dimensions
        intersection = (pred * target).sum(dim=(2, 3))
        sum_pred = pred.sum(dim=(2, 3))
        sum_target = target.sum(dim=(2, 3))
        
        # Compute Dice coefficient
        dice = (2. * intersection + smooth) / (sum_pred + sum_target + smooth)
        
        # Return the average Dice loss over the batch
        return 1 - dice.mean()

    def cross_entropy_loss(self): #cross entropy not used, as BCE is used instead
        def ce_loss(inputs, targets):
            inputs = inputs.contiguous()
            targets = targets.contiguous()
            probs = F.softmax(inputs, dim=1)  # Softmax to get probabilities for multi-class
            loss = -torch.sum(targets * torch.log(probs + 1e-5), dim=1)
            return loss.mean()
        return ce_loss


    def focal_loss(self, alpha=0.25, gamma=2):
        #Focal loss implementation for binary segmentation
        def fl_loss(inputs, targets):
            # Sigmoid already applied in the models
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-BCE_loss)  # Probability of being classified correctly
            loss = alpha * (1 - pt) ** gamma * BCE_loss
            return loss.mean()
        
        return fl_loss
    #Tversky loss function with variable alpha and beta
    def tversky_loss(self, alpha, beta, smooth=1e-5):
        def loss_fn(inputs, targets):
            true_pos = (inputs * targets).sum(dim=(1, 2, 3))
            false_neg = ((1 - inputs) * targets).sum(dim=(1, 2, 3))
            false_pos = (inputs * (1 - targets)).sum(dim=(1, 2, 3))

            tversky_index = (true_pos + smooth) / ( 
                true_pos + alpha * false_pos + beta * false_neg + smooth
            )
            return (1 - tversky_index).mean()

        return loss_fn


    def train(self):
        for epoch in range(self.config.epochs):
            # Training loop
            self.model.train()
            train_loss = 0
            for images, masks in self.train_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                
                # Handle multi-output models (HED and FPN)
                if isinstance(outputs, list):
                    loss = sum(self.loss_function(output, masks) for output in outputs) / len(outputs)
                else:
                    loss = self.loss_function(outputs, masks)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            # Calculate average training loss for the epoch
            avg_train_loss = train_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)

            # Validation loop
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_images, val_masks in self.val_loader:
                    val_images, val_masks = val_images.to(self.device), val_masks.to(self.device)
                    val_outputs = self.model(val_images)
                    
                    # Handle multi-output models (HED and FPN)
                    if isinstance(val_outputs, list):
                        val_loss += sum(self.loss_function(output, val_masks) for output in val_outputs) / len(val_outputs)
                    else:
                        val_loss += self.loss_function(val_outputs, val_masks)
                
                avg_val_loss = val_loss / len(self.val_loader)
                self.val_losses.append(avg_val_loss)
            
            # Print stats and check for improvement
            print(f"Epoch [{epoch + 1}/{self.config.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            should_stop = self._check_improvement(avg_val_loss)
            if should_stop:
                break

        # After training completes, plot the losses
        self.plot_losses()

    def _check_improvement(self, val_loss):
        #Checks improvement and increment patiance if no improvement is there
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            print("Model improved")
            torch.save(self.model.state_dict(), self.config.model_save_path)   #Save the latest best model
        else:
            self.epochs_without_improvement += 1
            print(f"No improvement in validation loss for {self.epochs_without_improvement} epochs.")
        
        if self.epochs_without_improvement >= self.config.patience:
            print(f"Early stopping after {self.epochs_without_improvement} epochs without improvement.")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
            print(f"Model saved to {self.model_save_path}")
            return True  # training should stop
        return False  # Continue training

    def plot_losses(self):
        #Plot training and validation losses over epochs

        # Convert all loss values to floats
        train_losses = [loss if isinstance(loss, float) else loss.cpu().item() for loss in self.train_losses]
        val_losses = [loss if isinstance(loss, float) else loss.cpu().item() for loss in self.val_losses]
        
        # save data
        np.savetxt(os.path.join(self.config.base_save_path, 'train_losses.csv'), train_losses, delimiter=",", header="Training Loss", comments="")
        np.savetxt(os.path.join(self.config.base_save_path, 'val_losses.csv'), val_losses, delimiter=",", header="Validation Loss", comments="")

        sns.set_theme()
        sns.set_style("darkgrid")
        sns.set_context("paper", font_scale=1.5) 

        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, len(train_losses) + 1),
            train_losses,
            label='Training Loss',
            color='tab:blue',
            marker='o',
            linestyle='-'
        )
        plt.plot(
            range(1, len(val_losses) + 1),
            val_losses,
            label='Validation Loss',
            color='tab:orange',
            marker='s',
            linestyle='--'
        )
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend(loc='upper right')
        plt.locator_params(axis='x', nbins=15)  # Increase number of ticks on x-axis
        plt.locator_params(axis='y', nbins=10)  # Increase number of ticks on y-axis
        plt.ylim([0.1, 0.9])
        
        # Save
        plt.tight_layout()
        plt.savefig(self.loss_save_path, format='png', dpi=300)
        plt.close()
        print(f"Loss plot saved to {self.loss_save_path}")