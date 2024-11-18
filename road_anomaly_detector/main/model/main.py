import torch
import argparse
from data_loader import get_data_loaders
from trainer import Trainer
from config.config import Config
from metrics.metrics import compute_metrics, hausdorff_distance_95, evaluate_model, run_inference, run_inference_on_folder
from sklearn.metrics import f1_score
from torchvision import transforms
from PIL import Image
import numpy as np


def main():
    #Clear vram
    torch.cuda.empty_cache()
    # Main parser
    parser = argparse.ArgumentParser(description="Road Anomaly Detection Framework")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Train subparser
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument('--model_name', type=str, default='UNet_advanced', help='Model to train: UNet_advanced, UNet_advanced')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    train_parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    train_parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    train_parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    train_parser.add_argument('--dataset_name', type=str, required=True, help="Dataset name")
    train_parser.add_argument('--loss_function', type=str, default='dice', help="Loss function: 'dice', 'bce', ce, focal")

    # Inference subparser
    inference_parser = subparsers.add_parser("inference", help="Run inference on a single image")
    inference_parser.add_argument('--model_name', type=str, default='UNet_advanced', help='Model for inference: UNet_advanced, UNet_advanced')
    inference_parser.add_argument('--image_path', type=str, required=True, help='Path to the image for inference')
    inference_parser.add_argument('--model_path', type=str, default='./model.pth', help='Path to the trained model')
    
    # Inference on folder subparser
    inference_folder_parser = subparsers.add_parser("inference_on_folder", help="Run inference on all images in a folder")
    inference_folder_parser.add_argument('--model_name', type=str, default='UNet_advanced', help='Model for inference: UNet_advanced, UNet_advanced')
    inference_folder_parser.add_argument('--folder_path', type=str, required=True, help='Path to the folder containing images for inference')
    inference_folder_parser.add_argument('--model_path', type=str, default='./model.pth', help='Path to the trained model')

    # Test subparser
    test_parser = subparsers.add_parser("test", help="Evaluate the model on the test set")
    test_parser.add_argument('--model_name', type=str, default='UNet_advanced', help='Model to test: UNet_advanced, UNet_advanced')
    test_parser.add_argument('--dataset_name', type=str, required=True, help="Data set name")
    test_parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    test_parser.add_argument('--model_path', type=str, default='./model.pth', help='Path to the trained model')

    args = parser.parse_args()

    # Load configuration
    config = Config(model_name=args.model_name)
    device = config.device

    if args.mode == "train":
        # Training mode
        train_loader, val_loader = get_data_loaders(
            dataset_name=args.dataset_name,
            batch_size=args.batch_size,
            image_size=(448, 448),
            test_size=0.2
        )

        trainer = Trainer(
            model=config.model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            loss_type=args.loss_function,  # Correct name here
            model_save_path='./model_files/model.pth',
            patience=args.patience,
        )
        trainer.train()
        print("Training complete! Model saved to './model.pth'.")

    elif args.mode == "inference":
        # Inference mode
        config = Config(model_name=args.model_name)  # Ensure the model architecture is properly set up
        config.model.load_state_dict(torch.load(args.model_path))  # Load state_dict
        config.model.to(device)

        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ])
        print(run_inference.__code__.co_varnames)

        run_inference(
        model=config.model,
        image_path=args.image_path,
        transform=transform,
        device=device,  # This should match the device you are using
        output_dir="./outputs"  # Ensure this is passed if the function expects it
    )
    elif args.mode == "inference_on_folder":
        # Inference on folder mode
        config = Config(model_name=args.model_name)  # Ensure the model architecture is properly set up
        config.model.load_state_dict(torch.load(args.model_path))  # Load state_dict
        config.model.to(device)

        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ])

        # Run inference on all images in the folder
        run_inference_on_folder(
            model=config.model,
            folder_path=args.folder_path,
            transform=transform,
            device=device,  # This should match the device you are using
            output_dir="./outputs"  # Ensure this is passed if the function expects it
        )

    elif args.mode == "test":
        # Test mode
        _, test_loader = get_data_loaders(
            dataset_name=args.dataset_name,
            batch_size=args.batch_size,
            image_size=(448, 448),
            test_size=0.1
        )

        config.model.load_state_dict(torch.load(args.model_path))
        config.model.to(device)

        evaluate_model(config.model, test_loader, device)

if __name__ == "__main__":
    main()
