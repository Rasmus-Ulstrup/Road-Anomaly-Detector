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
            return self.cross_entropy_loss()
        elif loss_type == "focal":
            alpha = kwargs.get('alpha', 0.995)  # Alpha to mittigate class imbalance
            gamma = kwargs.get('gamma', 2)  # Default gamma for focusing on hard examples
            return self.focal_loss(alpha, gamma)
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
            torch.save(self.model.state_dict(), "model.pth")        #Saves the latest best model
        else:
            self.epochs_without_improvement += 1
            print(f"No improvement in validation loss for {self.epochs_without_improvement} epochs.")
        
        if self.epochs_without_improvement >= self.patience:
            print(f"Early stopping after {self.epochs_without_improvement} epochs without improvement.")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
            print(f"Model saved to {self.model_save_path}")
            exit()
