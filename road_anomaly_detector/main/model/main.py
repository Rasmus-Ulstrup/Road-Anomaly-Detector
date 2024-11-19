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
    train_parser.add_argument('--model_name', type=str, default='UNet_advanced', help='Model to train')
    train_parser.add_argument('--dataset_name', type=str, required=True, help="Dataset name")
    train_parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    train_parser.add_argument('--image_size', type=tuple, default=(448, 448), help='Image size (height, width)')
    train_parser.add_argument('--test_size', type=float, default=0.2, help='Validation/test split size')
    train_parser.add_argument('--loss_function', type=str, default='dice', help="Loss function")
    train_parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    train_parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')


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

    # Initialize config with all parameters
    # config = Config(
    #     model_name=args.model_name,
    #     dataset_name=args.dataset_name,
    #     batch_size=args.batch_size,
    #     image_size=args.image_size,
    #     test_size=args.test_size,
    #     loss_function=args.loss_function,
    #     learning_rate=args.learning_rate,
    #     epochs=args.epochs,
    #     patience=args.patience,
    # )

    if args.mode == "train":
        config = Config(
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            batch_size=args.batch_size,
            image_size=args.image_size,
            test_size=args.test_size,
            loss_function=args.loss_function,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            patience=args.patience,
        )
        train_loader, val_loader = get_data_loaders(config)
        trainer = Trainer(model=config.model, train_loader=train_loader, val_loader=val_loader, config=config)
        trainer.train()
        
    elif args.mode == "inference":
        config = Config(model_name=args.model_name)  # Only pass parameters relevant for inference
        config.model.load_state_dict(torch.load(args.model_path))
        run_inference(config.model, args.image_path, device=config.device)
    elif args.mode == "inference_on_folder":
        config = Config(model_name=args.model_name)
        config.model.load_state_dict(torch.load(args.model_path))
        run_inference_on_folder(config.model, args.folder_path, device=config.device)

    elif args.mode == "test":
        config = Config(
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            batch_size=args.batch_size
        )
        _, test_loader = get_data_loaders(config)
        config.model.load_state_dict(torch.load(args.model_path))
        evaluate_model(config.model, test_loader, config.device)

if __name__ == "__main__":
    main()
