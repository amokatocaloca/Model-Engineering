import os
from feature_engineering import feature_engineering
from model_training import main_training
from eda import load_and_process_data

# Get **base directory** dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print("Starting the pipeline...")

    # Define relative file paths
    file_path = os.path.join(BASE_DIR, "data", "Data for Task 1.csv") 
    processed_path = os.path.join(BASE_DIR, "processed_data") 

    # Ensure processed directory exists
    os.makedirs(processed_path, exist_ok=True)

    # ✅ Load data only ONCE
    print("Loading and processing data...")
    data = load_and_process_data(file_path)

    if data is None:
        print("Pipeline aborted: Failed to load and process data.")
    else:
        print("Running feature engineering...")
        feature_engineering(data, processed_path)

        # Training
        print("Running model training...")
        main_training()
