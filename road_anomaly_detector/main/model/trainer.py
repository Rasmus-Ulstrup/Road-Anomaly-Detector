import torch
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, train_loader, val_loader, num_epochs=500, 
                 learning_rate=1e-3, patience=10, model_save_path="advanced_unet_segmentation_model.pth",
                 loss_type="dice", device=None, **kwargs):  # Align argument name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')  # Default to GPU if available
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.model_save_path = model_save_path
        
        # Select optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # Select loss function
        self.loss_function = self.select_loss_function(loss_type, **kwargs)
        
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
            return torch.nn.CrossEntropyLoss()
        elif loss_type == "focal":
            return self.focal_loss(**kwargs)
        else:
            raise ValueError(f"Loss function {loss_type} not supported")

    def dice_loss(self, pred, target, smooth=1e-5):
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        dice = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
        return 1 - dice.mean()

    def focal_loss(self, inputs, targets, alpha=0.25, gamma=2, smooth=1e-5):
        """
        Focal Loss implementation for binary classification
        Args:
            inputs: Predicted values (logits)
            targets: True values (labels)
            alpha: Balancing factor for class imbalance
            gamma: Focusing parameter to reduce loss for well-classified examples
            smooth: Small constant to avoid division by zero
        """
        inputs = inputs.contiguous()
        targets = targets.contiguous()
        
        # Apply sigmoid to the inputs
        inputs = torch.sigmoid(inputs)
        
        # Compute cross-entropy loss
        cross_entropy = -targets * torch.log(inputs + smooth) - (1 - targets) * torch.log(1 - inputs + smooth)
        
        # Focal loss adjustment
        loss = alpha * ((1 - inputs) ** gamma) * cross_entropy
        return loss.mean()

    def train(self):
        for epoch in range(self.num_epochs):
            # Training loop
            self.model.train()
            train_loss = 0
            for images, masks in self.train_loader:
                # Move both images and masks to the correct device
                images, masks = images.to(self.device), masks.to(self.device)
                    
                # Forward pass
                outputs = self.model(images)
                loss = self.loss_function(outputs, masks)
                    
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                    
                train_loss += loss.item()
            
            # Validation loop
            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for val_images, val_masks in self.val_loader:
                    # Move validation images and masks to the same device
                    val_images, val_masks = val_images.to(self.device), val_masks.to(self.device)
                    
                    val_outputs = self.model(val_images)
                    val_loss += self.loss_function(val_outputs, val_masks)
                val_loss /= len(self.val_loader)

            # Print stats and check for improvement
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss/len(self.train_loader)}, Val Loss: {val_loss:.4f}")
            self._check_improvement(val_loss)

    def _check_improvement(self, val_loss):
        """Check if the model improved and handle early stopping."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            print("Model improved")
            torch.save(self.model, self.model_save_path)
        else:
            self.epochs_without_improvement += 1
            print(f"No improvement in validation loss for {self.epochs_without_improvement} epochs.")
        
        if self.epochs_without_improvement >= self.patience:
            print(f"Early stopping after {self.epochs_without_improvement} epochs without improvement.")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
            print(f"Model saved to {self.model_save_path}")
            exit()