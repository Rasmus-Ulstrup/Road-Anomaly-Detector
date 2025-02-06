# run_multiple_trainings.py

import subprocess
import os
import yaml
from datetime import datetime
import csv

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_timestamped_log_directory(parent_dir="logs"):
    """Create a timestamped subdirectory under the parent logs directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(parent_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def sanitize_learning_rate(lr):
    """Remove the decimal point from the learning rate for filename purposes."""
    return str(lr).replace('.', '')

def construct_model_save_path(config, model_name, dataset_name, loss_function):
    """Construct the model save path based on the given naming convention."""
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    patience = config["patience"]
    epochs = config["epochs"]
    
    # Extract loss function name and parameters if any
    loss_name = loss_function["name"]
    if loss_name in ["tversky", "focal"]:
        loss_suffix = f"{loss_name}_alpha{loss_function['alpha']}_gamma{loss_function['gamma']}"
    else:
        loss_suffix = loss_name
    
    # Construct directory name
    dir_name = f"{model_name}_{dataset_name}_lr{sanitize_learning_rate(learning_rate)}_b{batch_size}_p{patience}_e{epochs}_{loss_suffix}"
    model_save_path = os.path.join("model_files", dir_name, "model.pth")
    
    return model_save_path

def construct_log_filename(action, config, loss_function):
    """Construct a log filename based on the action (train/test) and configuration."""
    model_name = config["model_name"]
    dataset_name = config["dataset_name"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    patience = config["patience"]
    epochs = config["epochs"]
    
    loss_name = loss_function["name"]
    if loss_name in ["tversky", "focal"]:
        loss_suffix = f"{loss_name}_alpha{loss_function['alpha']}_gamma{loss_function['gamma']}"
    else:
        loss_suffix = loss_name
    
    log_filename = f"{model_name}_{dataset_name}_lr{sanitize_learning_rate(learning_rate)}_b{batch_size}_p{patience}_e{epochs}_{loss_suffix}_{action}.log"
    return log_filename

def run_command(cmd, log_path):
    """Run a subprocess command and log its output."""
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        process.wait()
        return process.returncode, log_path

def run_training(config, log_dir):
    """Runs the training process with the given configuration."""
    cmd = [
        "python", "main.py", "train",
        "--model_name", config["model_name"],
        "--dataset_name", config["dataset_name"],
        "--batch_size", str(config["batch_size"]),
        "--image_size", str(config["image_size"]),  # Convert list to tuple string
        "--test_size", str(config["test_size"]),
        "--loss_function", config["loss_function"]["name"],
        "--learning_rate", str(config["learning_rate"]),
        "--epochs", str(config["epochs"]),
        "--patience", str(config["patience"])
    ]

    # Add alpha and gamma if they exist for the loss function
    if config["loss_function"]["name"] in ["tversky", "focal"]:
        if "alpha" in config["loss_function"]:
            cmd.extend(["--alpha", str(config["loss_function"]["alpha"])])
        if "gamma" in config["loss_function"]:
            cmd.extend(["--gamma", str(config["loss_function"]["gamma"])])

    # **Add Preprocessing Flag if Enabled**
    if config.get("preprocessing", False):
        cmd.append("--preprocessing")

    # Define log file name based on configuration
    log_filename = construct_log_filename("train", config, config["loss_function"])
    log_path = os.path.join(log_dir, log_filename)

    with open(log_path, "w") as log_file:
        print(f"Starting training: {config}")
        process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        process.wait()  # Wait for the process to complete
        if process.returncode == 0:
            print(f"Training completed successfully: {config}")
        else:
            print(f"Training failed for configuration: {config}. Check log: {log_path}")

def run_testing(config, log_dir, model_save_path):
    """Runs the testing process with the given configuration and model path."""
    cmd = [
        "python", "main.py", "test",
        "--model_name", config["model_name"],
        "--dataset_name", config["dataset_name"],  # Assuming test uses the same dataset; adjust if different
        "--batch_size", str(config["batch_size"]),
        "--model_path", model_save_path

    ]

    # **Add Preprocessing Flag if Enabled for Testing (Optional)**
    if config.get("preprocessing", False):
        cmd.append("--preprocessing")

    # Define log file name based on configuration
    log_filename = construct_log_filename("test", config, config["loss_function"])
    log_path = os.path.join(log_dir, log_filename)

    with open(log_path, "w") as log_file:
        print(f"Starting testing: {config}")
        process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        process.wait()  # Wait for the process to complete
        if process.returncode == 0:
            print(f"Testing completed successfully: {config}")
        else:
            print(f"Testing failed for configuration: {config}. Check log: {log_path}")

def main_traning():
    # Load the YAML configuration
    config = load_config("config.yaml")

    # Ensure logs directory exists and create a timestamped subdirectory
    log_dir = create_timestamped_log_directory("logs")
    print(f"Logs will be saved to: {log_dir}")

    # Generate all combinations of datasets, models, and loss functions
    datasets = config.get("datasets", {})
    models = config.get("models", [])
    loss_functions = config.get("loss_functions", [])

    if not datasets:
        print("No datasets found in configuration. Exiting.")
        return
    if not models:
        print("No models found in configuration. Exiting.")
        return
    if not loss_functions:
        print("No loss functions found in configuration. Exiting.")
        return

    # Create 'model_files' directory if it doesn't exist
    os.makedirs("model_files", exist_ok=True)

    # Iterate through each combination
    for dataset_name, dataset_params in datasets.items():
        for model_name in models:
            for loss_function in loss_functions:
                # Prepare the configuration for this combination
                combination_config = {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "learning_rate": dataset_params.get("learning_rate", config.get("global", {}).get("learning_rate", 0.001)),
                    "batch_size": dataset_params.get("batch_size", config.get("global", {}).get("batch_size", 4)),
                    "loss_function": loss_function,
                    "epochs": dataset_params.get("epochs", config.get("global", {}).get("epochs", 50)),
                    "patience": dataset_params.get("patience", config.get("global", {}).get("patience", 10)),
                    "image_size": dataset_params.get("image_size", config.get("global", {}).get("image_size", [512, 512])),
                    "test_size": dataset_params.get("test_size", config.get("global", {}).get("test_size", 0.2)),
                    "preprocessing": dataset_params.get("preprocessing", config.get("global", {}).get("preprocessing", False))  # **Include Preprocessing Flag**
                }

                # Define the model save path based on the naming convention
                model_save_path = construct_model_save_path(
                    combination_config,
                    model_name,
                    dataset_name,
                    loss_function
                )

                # Ensure the directory for the model exists
                model_dir = os.path.dirname(model_save_path)
                os.makedirs(model_dir, exist_ok=True)

                # Run the training
                run_training(combination_config, log_dir)

                # After training, run the testing
                if os.path.exists(model_save_path):
                    run_testing(combination_config, log_dir, model_save_path)
                else:
                    print(f"Model file not found at {model_save_path}. Skipping testing for this configuration.")

def run_testing_tiles(config, log_dir, model_path):
    """Runs the tiles testing process with the given configuration."""
    cmd = [
        "python", "main.py", "tiles_test",
        "--model_name", config["model_name"],
        "--model_path", model_path,
        "--folder_path", config["folder_path"]
    ]

    # Add preprocessing flag if specified in the configuration
    if config.get("preprocessing", False):
        cmd.append("--preprocessing")

    # Define log file name based on configuration
    log_filename = construct_log_filename("tiles_test", config, config["loss_function"])
    log_path = os.path.join(log_dir, log_filename)

    # Execute the command and log its output
    with open(log_path, "w") as log_file:
        print(f"Starting tiles testing: {config}")
        process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        process.wait()  # Wait for the process to complete

        # Check the result and provide feedback
        if process.returncode == 0:
            print(f"Tiles testing completed successfully: {config}")
        else:
            print(f"Tiles testing failed: {config}. Check log: {log_path}")

    # Initialize a dictionary to store metrics
    metrics = {}

    # Search for all metrics!
    with open(log_path, "r") as log_file:
        for line in log_file:
            # Check for each metric and extract the value
            if "Correctness:" in line:
                try:
                    metrics["Correctness"] = float(line.split("Correctness:")[1].strip())
                except ValueError:
                    print("Could not parse Correctness value.")
            elif "Completeness:" in line:
                try:
                    metrics["Completeness"] = float(line.split("Completeness:")[1].strip())
                except ValueError:
                    print("Could not parse Completeness value.")
            elif "Quality:" in line:
                try:
                    metrics["Quality"] = float(line.split("Quality:")[1].strip())
                except ValueError:
                    print("Could not parse Quality value.")
            elif "F1:" in line:
                try:
                    metrics["F1"] = float(line.split("F1:")[1].strip())
                except ValueError:
                    print("Could not parse F1 value.")
            elif "Precision:" in line:
                try:
                    metrics["Precision"] = float(line.split("Precision:")[1].strip())
                except ValueError:
                    print("Could not parse Precision value.")
            elif "Recall:" in line:
                try:
                    metrics["Recall"] = float(line.split("Recall:")[1].strip())
                except ValueError:
                    print("Could not parse Recall value.")
            elif "Iou:" in line:
                try:
                    metrics["Iou"] = float(line.split("Iou:")[1].strip())
                except ValueError:
                    print("Could not parse IoU value.")

    # Print the metrics dictionary
    print("Metrics:", metrics)
    return metrics

def main_tiles_test(folder_path, model_path, preprocessing):
    # Load the YAML configuration
    config = load_config("config.yaml")

    # Generate all combinations of datasets, models, and loss functions
    datasets = config.get("datasets", {})
    models = config.get("models", [])
    loss_functions = config.get("loss_functions", [])

    if not datasets:
        print("No datasets found in configuration. Exiting.")
        return
    if not models:
        print("No models found in configuration. Exiting.")
        return
    if not loss_functions:
        print("No loss functions found in configuration. Exiting.")
        return
    
    os.makedirs("labeled_test", exist_ok=True)

    # Prepare the CSV file
    full_path = os.path.expanduser(folder_path)
    folder_name = os.path.basename(os.path.normpath(full_path))
    csv_file = f"./labeled_test/metrics_{folder_name}_Pre-{preprocessing}.csv"
    print(csv_file)

    # Ensure logs directory exists and create a timestamped subdirectory
    log_dir = create_timestamped_log_directory("logs")
    parent_folder = f"{folder_name}_Pre-{preprocessing}"
    log_dir = os.path.join(log_dir,parent_folder)
    print(f"Logs will be saved to: {log_dir}")
    os.makedirs(log_dir, exist_ok=True)

    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["Model Name", "Correctness", "Completeness", "Quality", "Precision", "Recall", "Iou", "F1"])

        # Iterate through each combination
        for dataset_name, dataset_params in datasets.items():
            for model_name in models:
                for loss_function in loss_functions:
                    # Prepare the configuration for this combination
                    combination_config = {
                        "dataset_name": dataset_name,
                        "model_name": model_name,
                        "learning_rate": dataset_params.get("learning_rate", config.get("global", {}).get("learning_rate", 0.001)),
                        "batch_size": dataset_params.get("batch_size", config.get("global", {}).get("batch_size", 4)),
                        "loss_function": loss_function,
                        "epochs": dataset_params.get("epochs", config.get("global", {}).get("epochs", 50)),
                        "patience": dataset_params.get("patience", config.get("global", {}).get("patience", 10)),
                        "image_size": dataset_params.get("image_size", config.get("global", {}).get("image_size", [512, 512])),
                        "test_size": dataset_params.get("test_size", config.get("global", {}).get("test_size", 0.2)),
                        "preprocessing": preprocessing,
                        "folder_path": folder_path
                    }

                    # Define the model save path based on the naming convention
                    model_save_path = construct_model_save_path(
                        combination_config,
                        model_name,
                        dataset_name,
                        loss_function
                    )

                    # Ensure the directory for the model exists
                    model_dir = os.path.dirname(model_save_path)
                    real_model_name = model_dir.replace("model_files/", "")
                    real_model_dir = os.path.join(model_path, real_model_name, "model.pth")


                    # Run testing and get metrics
                    metrics = run_testing_tiles(combination_config, log_dir, real_model_dir)

                    # Convert keys to lowercase for consistency (optional)
                    metrics = {k.lower(): v for k, v in metrics.items()}

                    # Write the data row to the CSV, with a fallback if a key is missing
                    if metrics:
                        writer.writerow([
                            real_model_name,
                            metrics.get("correctness", "N/A"),
                            metrics.get("completeness", "N/A"),
                            metrics.get("quality", "N/A"),
                            metrics.get("precision", "N/A"),
                            metrics.get("recall", "N/A"),
                            metrics.get("iou", "N/A"),
                            metrics.get("f1", "N/A")
                        ])
                    print(f"Saved metrics for {real_model_name} to {csv_file}")

def main_testing(model_path):
    # Load the YAML configuration
    config = load_config("config.yaml")

    # Ensure logs directory exists and create a timestamped subdirectory
    log_dir = create_timestamped_log_directory("logs")
    print(f"Logs will be saved to: {log_dir}")

    # Generate all combinations of datasets, models, and loss functions
    datasets = config.get("datasets", {})
    models = config.get("models", [])
    loss_functions = config.get("loss_functions", [])

    if not datasets:
        print("No datasets found in configuration. Exiting.")
        return
    if not models:
        print("No models found in configuration. Exiting.")
        return
    if not loss_functions:
        print("No loss functions found in configuration. Exiting.")
        return

    # Create 'model_files' directory if it doesn't exist
    os.makedirs("model_files", exist_ok=True)

    # Iterate through each combination
    for dataset_name, dataset_params in datasets.items():
        for model_name in models:
            for loss_function in loss_functions:
                # Prepare the configuration for this combination
                combination_config = {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "learning_rate": dataset_params.get("learning_rate", config.get("global", {}).get("learning_rate", 0.001)),
                    "batch_size": dataset_params.get("batch_size", config.get("global", {}).get("batch_size", 4)),
                    "loss_function": loss_function,
                    "epochs": dataset_params.get("epochs", config.get("global", {}).get("epochs", 50)),
                    "patience": dataset_params.get("patience", config.get("global", {}).get("patience", 10)),
                    "image_size": dataset_params.get("image_size", config.get("global", {}).get("image_size", [512, 512])),
                    "test_size": dataset_params.get("test_size", config.get("global", {}).get("test_size", 0.2)),
                    "preprocessing": dataset_params.get("preprocessing", config.get("global", {}).get("preprocessing", False))  # **Include Preprocessing Flag**
                }

                # Define the model save path based on the naming convention
                model_save_path = construct_model_save_path(
                    combination_config,
                    model_name,
                    dataset_name,
                    loss_function
                )

                # Ensure the directory for the model exists
                model_dir = os.path.dirname(model_save_path)
                real_model_name = model_dir.replace("model_files/", "")
                real_model_dir = os.path.join(model_path, real_model_name, "model.pth")
                print(real_model_dir)
                # Run the testing
                run_testing(combination_config, log_dir, real_model_dir)

                # After training, run the testing
                # if os.path.exists(model_save_path):
                #     run_testing(combination_config, log_dir, model_save_path)
                # else:
                #     print(f"Model file not found at {model_save_path}. Skipping testing for this configuration.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Road Anomaly Detection")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train", help="Train the model")
    test_parser = subparsers.add_parser("test", help="Evaluate the model on the tiles test set")
    test2_parser = subparsers.add_parser("test2", help="Evaluate the model on the test set (dataset)")
    args = parser.parse_args()
    
    if args.mode == "train":
        print("Now training!")
        main_traning()
    elif args.mode == "test":
        print("Now testing!")

        folder_paths = [
            "~/Documents/test_data/images_sti_test/",
            "~/Documents/test_data/images_vej_test/",
            "~/Documents/test_data/combined_test/"
        ]

        model_paths = [
            "~/models_preprocessed/",
            "~/model_no_preprocessed/"
            # "model_files/"
        ]

        preprocessing_options = [True, False]
        # preprocessing_options = [True, False]
        # Generate and run all combinations
        for folder_path in folder_paths:
            for model_path in model_paths:
                # Apply constraints on preprocessing based on the model path
                # if model_path == "model_files/":
                #     print("test")
                #     preprocessing = True
                if model_path == "~/models_preprocessed/":
                    preprocessing = True
                elif model_path == "~/model_no_preprocessed/":
                    preprocessing = False
                else:
                    continue  # Skip invalid combinations
                # if preprocessing:
                #     continue
                folder_path = os.path.expanduser(folder_path)
                model_path = os.path.expanduser(model_path)
                main_tiles_test(folder_path, model_path, preprocessing)

    elif args.mode == "test2":
        print("Now testing on test dataset!")

        folder_paths = [
            "~/Documents/testsetz"
        ]

        model_paths = [
            "~/models_preprocessed/",
            #"~/model_no_preprocessed/"
            # "model_files/"
        ]

        preprocessing_options = [True, False]
        # preprocessing_options = [True, False]
        # Generate and run all combinations
        for folder_path in folder_paths:
            for model_path in model_paths:
                # Apply constraints on preprocessing based on the model path
                # if model_path == "model_files/":
                #     print("test")
                #     preprocessing = True
                if model_path == "~/models_preprocessed/":
                    preprocessing = True
                elif model_path == "~/model_no_preprocessed/":
                    preprocessing = False
                else:
                    continue  # Skip invalid combinations
                # if preprocessing:
                #     continue
                folder_path = os.path.expanduser(folder_path)
                model_path = os.path.expanduser(model_path)
                print(f"Folder path: {folder_path}      model_path: {model_path}")
                main_testing(model_path)