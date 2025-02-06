import os
import pandas as pd

def extract_metrics(model_files_path, output_csv='compiled_metrics.csv'):
    """
    Extracts metrics from each model's metrics.csv, identifies the best models for each metric,
    and compiles all information into a single CSV file.

    :param model_files_path: Path to the directory containing all model folders.
    :param output_csv: Name of the output CSV file.
    """
    # List to store all metrics
    all_metrics = []

    # Iterate through each item in the model_files directory
    for model_name in os.listdir(model_files_path):
        model_path = os.path.join(model_files_path, model_name)
        
        # Check if it's a directory
        if os.path.isdir(model_path):
            metrics_file = os.path.join(model_path, 'metrics.csv')
            
            # Check if metrics.csv exists
            if os.path.isfile(metrics_file):
                try:
                    # Read the metrics.csv
                    df = pd.read_csv(metrics_file)
                    
                    # Initialize a dictionary to store metrics
                    metrics_dict = {'Model': model_name}
                    
                    # Extract each required metric
                    for _, row in df.iterrows():
                        metric_name = row['Metric']
                        metric_value = row['Average']
                        if metric_name in ['Correctness', 'Completeness', 'Quality', 'F1']:
                            metrics_dict[metric_name] = metric_value
                            
                    # Ensure all metrics are present
                    required_metrics = ['Correctness', 'Completeness', 'Quality', 'F1']
                    if all(metric in metrics_dict for metric in required_metrics):
                        all_metrics.append(metrics_dict)
                    else:
                        missing = [m for m in required_metrics if m not in metrics_dict]
                        print(f"Warning: Missing metrics {missing} in {metrics_file}. Skipping this model.")
                
                except Exception as e:
                    print(f"Error reading {metrics_file}: {e}")
            else:
                print(f"Warning: {metrics_file} does not exist. Skipping this model.")

    if not all_metrics:
        print("No metrics were compiled. Please check the metrics.csv files and try again.")
        return

    # Create a DataFrame from the list of metrics
    compiled_df = pd.DataFrame(all_metrics)

    # Identify the best model for each metric
    for metric in ['Correctness', 'Completeness', 'Quality', 'F1']:
        best_value = compiled_df[metric].max()
        compiled_df[f'Best_{metric}'] = compiled_df[metric] == best_value

    # Optionally, sort the DataFrame by F1 in descending order
    compiled_df.sort_values(by='F1', ascending=False, inplace=True)

    # Reorder columns for better readability
    ordered_columns = [
        'Model', 
        'Correctness', 'Best_Correctness',
        'Completeness', 'Best_Completeness',
        'Quality', 'Best_Quality',
        'F1', 'Best_F1'
    ]
    compiled_df = compiled_df[ordered_columns]

    # Save the compiled metrics to a CSV file
    output_path = os.path.join(model_files_path, output_csv)
    compiled_df.to_csv(output_path, index=False)

    print(f"Compiled metrics have been saved to {output_path}\n")

    # Display Best Models per Metric
    print("Best Models per Metric:")
    for metric in ['Correctness', 'Completeness', 'Quality', 'F1']:
        best_models = compiled_df.loc[compiled_df[f'Best_{metric}'], 'Model'].values
        best_value = compiled_df[metric].max()
        print(f" - {metric}: {', '.join(best_models)} with value {best_value}")

if __name__ == "__main__":
    # Replace this path with your actual model_files directory path
    # model_files_directory = '/home/crack/Road-Anomaly-Detector/road_anomaly_detector/main/model/model_files'
    model_files_directory = '/home/crack/models_preprocessed'
    # model_files_directory = '/home/crack/model_no_preprocessed'
    # Call the function to extract and compile metrics
    extract_metrics(model_files_directory)
