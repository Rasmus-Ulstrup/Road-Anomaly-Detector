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
from utils.tiles import run_main_tiles


def main():
    #Clear vram
    torch.cuda.empty_cache()
    # Main parser
    parser = argparse.ArgumentParser(description="Road Anomaly Detection Framework")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Train subparser
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument('--model_name', type=str, default='UNet', help='Model to train')
    train_parser.add_argument('--dataset_name', type=str, required=True, help="Dataset name")
    train_parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    def parse_tuple(value):
        try:
            return tuple(map(int, value.strip('()').split(',')))
        except ValueError:
            raise argparse.ArgumentTypeError("Image size must be a tuple of two integers, e.g., '(512, 512)'")
    train_parser.add_argument('--image_size', type=parse_tuple, default="512, 512", help='Image size (height, width)')
    train_parser.add_argument('--test_size', type=float, default=0.2, help='Validation/test split size')
    train_parser.add_argument('--loss_function', type=str, default='dice', help="Loss function")
    train_parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    train_parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    train_parser.add_argument('--argumentation', type=int, default=True, help='Argumentation 1 or 0')
    train_parser.add_argument('--alpha', type=float, default={}, help='Additional loss parameters')
    train_parser.add_argument('--gamma', type=float, default={}, help='Additional loss parameters GAMMA OR BETA')


    # Inference subparser
    inference_parser = subparsers.add_parser("inference", help="Run inference on a single image")
    inference_parser.add_argument('--model_name', type=str, default='unet', help='Model for inference: unet, unet')
    inference_parser.add_argument('--image_path', type=str, required=True, help='Path to the image for inference')
    inference_parser.add_argument('--model_path', type=str, default='./model.pth', help='Path to the trained model')
    inference_parser.add_argument('--output_dir', type=str, default='./output/inference_single', help='Path to output place')
    
    # Inference on folder subparser
    inference_folder_parser = subparsers.add_parser("inference_on_folder", help="Run inference on all images in a folder")
    inference_folder_parser.add_argument('--model_name', type=str, default='unet', help='Model for inference: unet, unet')
    inference_folder_parser.add_argument('--folder_path', type=str, required=True, help='Path to the folder containing images for inference')
    inference_folder_parser.add_argument('--model_path', type=str, default='./model.pth', help='Path to the trained model')
    inference_folder_parser.add_argument('--output_dir', type=str, default='./output/inference_folder', help='Path to output place')
    inference_folder_parser.add_argument('--image_size', type=parse_tuple, default=(512, 512), help='Image size (height, width)')

    # Test subparser
    test_parser = subparsers.add_parser("test", help="Evaluate the model on the test set")
    test_parser.add_argument('--model_name', type=str, default='unet', help='Model to test: unet, unet')
    test_parser.add_argument('--dataset_name', type=str, required=True, help="Data set name")
    test_parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    test_parser.add_argument('--model_path', type=str, required=True, default='model_files/.model.pth', help='Path to the trained model')

    # Tiles subparser
    tiles_parser = subparsers.add_parser("tiles", help="Splits image into tiles and runs inference")
    tiles_parser.add_argument('--model_name', type=str, default='unet', help='Model for inference: unet, unet')
    tiles_parser.add_argument('--model_path', type=str, default='./model_files/UNet_cfd_lr0001_b4_p20_e500_dice.pth', help='Path to the trained model')
    tiles_parser.add_argument('--image_size', type=parse_tuple, default=(512, 512), help='Image size (height, width)')
    tiles_parser.add_argument('--output_dir', type=str, default='./output/tiles', help='Path to output place')
    tiles_parser.add_argument('--folder_path', type=str, required=True, help='Path to the folder containing images for inference')
    tiles_parser.add_argument('--overlap', type=str, default=50, help='the overlap between images')
    tiles_parser.add_argument('--max_workers', type=str, default=4, help='How many threads to run this')
    #Overlap

    args = parser.parse_args()

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
            Argumentation=args.argumentation,
            alpha=args.alpha,
            gamma=args.gamma
        )
        config.update_trainer_save_paths()
        train_loader, val_loader,_ = get_data_loaders(config)
        trainer = Trainer(model=config.model, train_loader=train_loader, val_loader=val_loader, config=config)
        trainer.train()
        
    elif args.mode == "inference":
        config = Config(model_name=args.model_name)  # Only pass parameters relevant for inference
        config.model.load_state_dict(torch.load(args.model_path)) #  weights_only=True ?
        run_inference(config,config.model, args.image_path, device=config.device, output_dir=args.output_dir)

    elif args.mode == "inference_on_folder":
        config = Config(model_name=args.model_name, image_size=args.image_size)
        config.model.load_state_dict(torch.load(args.model_path)) #  weights_only=True ?
        run_inference_on_folder(config, config.model, args.folder_path, device=config.device, output_dir=args.output_dir)

    elif args.mode == "test":
        config = Config(
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            batch_size=args.batch_size
        )
        config.update_metric_save_path(args.model_path)
        _,_, test_loader = get_data_loaders(config)
        config.model.load_state_dict(torch.load(args.model_path))
        evaluate_model(config, config.model, test_loader, config.device)

    elif args.mode == 'tiles':
        config = Config(model_name=args.model_name, image_size=args.image_size)
        image_path = '/home/crackscope/Road-Anomaly-Detector/road_anomaly_detector/main/model/utils/test_image.png'
        
        if config.image_size[0] != config.image_size[1]:
            raise Exception("Image size needs to be a square")

        config.model.load_state_dict(torch.load(args.model_path)) #  weights_only=True ?

        run_main_tiles(
            config, 
            image_dir=args.folder_path, 
            output_dir=args.output_dir,
            model=config.model, 
            device=config.device, 
            tile_size=config.image_size[1],
            overlap=args.overlap,
            max_workers=args.max_workers
        )




if __name__ == "__main__":
    main()
