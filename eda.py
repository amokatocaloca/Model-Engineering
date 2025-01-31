import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_and_process_data(file_path):
    print(f"Loading dataset from: {file_path}")

    # Ensure file path is relative
    file_path = os.path.join(BASE_DIR, file_path)

    # Load dataset with error handling
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{file_path}' is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Error: The file '{file_path}' could not be parsed correctly. Check the format.")
        return None

    # Ensure dataset is not empty
    if data is None or data.empty:
        print("Error: Loaded dataset is empty.")
        return None

    print(f"Dataset loaded successfully. Shape: {data.shape}")

    # Create relative results directory
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Drop unnecessary columns
    if 'Unnamed: 32' in data.columns:
        data = data.drop(columns=['Unnamed: 32'])

    # Check for missing values
    missing_values = data.isnull().sum()
    print(f"Missing values:\n{missing_values[missing_values > 0]}")

    # Class distribution
    if 'diagnosis' not in data.columns:
        print("Error: 'diagnosis' column is missing in the dataset.")
        return None

    class_counts = data['diagnosis'].value_counts()
    print("Class Distribution:")
    print(class_counts)

    # Save Class Distribution Plot
    plt.figure(figsize=(6, 4))
    sns.countplot(data=data, x='diagnosis', order=class_counts.index)
    plt.title('Class Distribution of Tumor Diagnosis', pad=15)
    plt.xlabel('Diagnosis (B=Benign, M=Malignant)', labelpad=10)
    plt.ylabel('Count', labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "class_distribution.png"))
    plt.close('all')

    # Feature Distributions: Histograms
    data_numeric = data.select_dtypes(include=['float64', 'int64']).drop(columns=['id'], errors='ignore')

    data_numeric.hist(bins=15, figsize=(15, 15))
    plt.suptitle('Feature Distributions', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "feature_distributions.png"))
    plt.close('all')

    # Box plot for Outlier Detection
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=data_numeric, orient='h')
    plt.title('Box Plots of Features to Identify Outliers', pad=15)
    plt.xlabel('Feature Value', labelpad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "box_plots_outliers.png"))
    plt.close('all')

    # Correlation Heatmap
    plt.figure(figsize=(15, 15))
    correlation_matrix = data_numeric.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', square=True, linewidths=0.5)
    plt.title('Correlation Heatmap of Features', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "correlation_heatmap.png"))
    plt.close('all')

    return data
