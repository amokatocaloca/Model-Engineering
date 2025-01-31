import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

def feature_engineering(data, processed_path):
    print("Feature Engineering started...")

    # Encode target variable ('diagnosis') into numeric format
    label_encoder = LabelEncoder()
    data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])  # B=0, M=1

    # Drop standard error (_se) features as per analysis
    se_columns = [col for col in data.columns if '_se' in col]
    data = data.drop(columns=se_columns, errors='ignore')
    print(f"Dropped standard error features: {se_columns}")

    # Define features and target
    X = data.drop(columns=['id', 'diagnosis'], errors='ignore')
    y = data['diagnosis']

    print("Splitting data into training & testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Balancing classes using SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("Removing highly correlated features...")
    # Compute correlation matrix & remove one feature per correlated group (correlation > 0.90)
    corr_matrix = X_train_resampled.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_keep = [column for column in upper.columns if not any(upper[column] > 0.90)]
    
    print(f"Keeping these features after correlation filtering: {to_keep}")
    X_train_resampled = X_train_resampled[to_keep]
    X_test = X_test[to_keep]

    print("Scaling data using MinMaxScaler...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame with column names
    X_train_df = pd.DataFrame(X_train_scaled, columns=X_train_resampled.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X_train_resampled.columns)
    y_train_df = pd.DataFrame(y_train_resampled, columns=['diagnosis'])
    y_test_df = pd.DataFrame(y_test, columns=['diagnosis'])

    # Ensure processed path exists
    os.makedirs(processed_path, exist_ok=True)

    print("Saving processed datasets...")
    X_train_df.to_csv(os.path.join(processed_path, 'X_train.csv'), index=False)
    X_test_df.to_csv(os.path.join(processed_path, 'X_test.csv'), index=False)
    y_train_df.to_csv(os.path.join(processed_path, 'y_train.csv'), index=False)
    y_test_df.to_csv(os.path.join(processed_path, 'y_test.csv'), index=False)

    print("Feature engineering completed. Processed datasets saved.")
