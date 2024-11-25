import torch
import torch.nn.functional as F
from config.config import Config

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.config = config  # Correctly assign the passed config object
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.device  # Get device from the config object
        self.num_epochs = config.epochs
        self.learning_rate = config.learning_rate
        self.patience = config.patience
        self.model_save_path = config.model_save_path

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Select the loss function
        self.loss_function = self.select_loss_function(config.loss_function)
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

    def select_loss_function(self, loss_type, **kwargs):
        """Select the loss function based on the string type."""
        if loss_type == "dice":
            return self.dice_loss
        elif loss_type == "bce":
            # Set alpha to 0.9 by default if it's not passed in kwargs
            pos_weight = kwargs.get('pos_weight', 0.9)  # Default to 0.9
            pos_weight_tensor = torch.tensor([pos_weight]).to(self.device)  # Move pos_weight to the correct device
            return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        elif loss_type == "ce":
            return self.cross_entropy_loss()
        elif loss_type == "focal":
            alpha = kwargs.get('alpha', 0.995)  # Alpha to mittigate class imbalance
            gamma = kwargs.get('gamma', 2)  # Default gamma for focusing on hard examples
            return self.focal_loss(alpha, gamma)
        elif loss_type == "tversky":
            alpha = kwargs.get('alpha', 0.3)  # Default alpha
            beta = kwargs.get('beta', 0.7)   # Default beta
            return self.tversky_loss(alpha, beta)
        else:
            raise ValueError(f"Loss function {loss_type} not supported")

    def dice_loss(self, pred, target, smooth=1e-5):
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        dice = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
        return 1 - dice.mean()

    def cross_entropy_loss(self):
        """Cross-Entropy Loss for multi-class segmentation."""
        def ce_loss(inputs, targets):
            inputs = inputs.contiguous()
            targets = targets.contiguous()
            probs = F.softmax(inputs, dim=1)  # Softmax to get probabilities for multi-class
            loss = -torch.sum(targets * torch.log(probs + 1e-5), dim=1)
            return loss.mean()
        return ce_loss


    def focal_loss(self, alpha, gamma):
        """Focal loss implementation for binary segmentation."""
        def fl_loss(inputs, targets):
            inputs = inputs.contiguous()
            targets = targets.contiguous()
            
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            pt = torch.exp(-BCE_loss)  # Probability of being classified correctly
            loss = alpha * (1 - pt) ** gamma * BCE_loss
            return loss.mean()

        return fl_loss

    def tversky_loss(self, alpha, beta, smooth=1e-5):
        def loss_fn(inputs, targets):
            inputs = torch.sigmoid(inputs)  # Apply sigmoid for probabilities
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
                
                # Handle multi-output models
                if isinstance(outputs, list):
                    loss = sum(self.loss_function(output, masks) for output in outputs) / len(outputs)
                else:
                    loss = self.loss_function(outputs, masks)
                
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            # Validation loop
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_images, val_masks in self.val_loader:
                    val_images, val_masks = val_images.to(self.device), val_masks.to(self.device)
                    val_outputs = self.model(val_images)
                    
                    # Handle multi-output models
                    if isinstance(val_outputs, list):
                        val_loss += sum(self.loss_function(output, val_masks) for output in val_outputs) / len(val_outputs)
                    else:
                        val_loss += self.loss_function(val_outputs, val_masks)
                
                val_loss /= len(self.val_loader)
            
            # Print stats and check for improvement
            print(f"Epoch [{epoch + 1}/{self.config.epochs}], Train Loss: {train_loss / len(self.train_loader):.4f}, Val Loss: {val_loss:.4f}")
            self._check_improvement(val_loss)

    def _check_improvement(self, val_loss):
        """Check if the model improved and handle early stopping."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            print("Model improved")
            torch.save(self.model.state_dict(), self.config.model_save_path)        #Saves the latest best model
        else:
            self.epochs_without_improvement += 1
            print(f"No improvement in validation loss for {self.epochs_without_improvement} epochs.")
        
        if self.epochs_without_improvement >= self.config.patience:
            print(f"Early stopping after {self.epochs_without_improvement} epochs without improvement.")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
            print(f"Model saved to {self.model_save_path}")
            exit()
