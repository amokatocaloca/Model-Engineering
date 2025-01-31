import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, precision_recall_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.inspection import permutation_importance

# Automatically detect the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
EVALUATION_DIR = os.path.join(BASE_DIR, "evaluation")

# Ensure the evaluation directory exists
os.makedirs(EVALUATION_DIR, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {model_name}')
    
    # Save the plot
    plt.savefig(os.path.join(EVALUATION_DIR, f"confusion_matrix_{model_name}.png"))
    plt.close('all')

def plot_feature_importance(model, feature_names, model_name, X_test, y_test):
    if model_name == "Logistic Regression":
        importance = np.abs(model.coef_[0])
    elif model_name == "Gradient Boosting":
        perm_importance = permutation_importance(model, X_test, y_test, scoring="accuracy", n_repeats=10, random_state=42)
        importance = perm_importance.importances_mean
    else:
        return  

    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title(f'Feature Importance for {model_name}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')

    # Save the plot
    plt.savefig(os.path.join(EVALUATION_DIR, f"feature_importance_{model_name}.png"))
    plt.close('all')

def main_training():
    # Load preprocessed data with relative paths
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_train.csv'))['diagnosis']
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_test.csv'))['diagnosis']

    feature_names = X_train.columns
    results = []

    def train_knn():
        param_grid = {'n_neighbors': [5, 7, 9, 11, 13], 'weights': ['distance']}
        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='recall')
        grid_search.fit(X_train, y_train)
        best_knn = grid_search.best_estimator_
        y_pred_knn = best_knn.predict(X_test)

        results.append({
            'Algorithm': 'KNN',
            'Best Hyperparameters': grid_search.best_params_,
            'Accuracy': accuracy_score(y_test, y_pred_knn),
            'Precision': precision_score(y_test, y_pred_knn),
            'Recall': recall_score(y_test, y_pred_knn),
            'F1 Score': f1_score(y_test, y_pred_knn)
        })

        plot_confusion_matrix(y_test, y_pred_knn, "KNN")

    def train_logistic_regression():
        param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
        grid_search = GridSearchCV(LogisticRegression(max_iter=500), param_grid, cv=5, scoring='recall')
        grid_search.fit(X_train, y_train)

        best_log_reg = grid_search.best_estimator_
        y_probs = best_log_reg.predict_proba(X_test)[:, 1]

        precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
        f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
        best_threshold = thresholds[np.argmax(f1_scores)]

        print(f"Optimal Decision Threshold for Logistic Regression: {best_threshold}")
        y_pred_adjusted = (y_probs >= best_threshold).astype(int)

        results.append({
            'Algorithm': 'Logistic Regression',
            'Best Hyperparameters': grid_search.best_params_,
            'Optimal Threshold': best_threshold,
            'Accuracy': accuracy_score(y_test, y_pred_adjusted),
            'Precision': precision_score(y_test, y_pred_adjusted),
            'Recall': recall_score(y_test, y_pred_adjusted),
            'F1 Score': f1_score(y_test, y_pred_adjusted)
        })

        plot_confusion_matrix(y_test, y_pred_adjusted, "Logistic Regression")
        plot_feature_importance(best_log_reg, feature_names, "Logistic Regression", X_test, y_test)

    def train_gradient_boosting():
        param_grid = {'max_iter': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
        grid_search = GridSearchCV(HistGradientBoostingClassifier(random_state=42), param_grid, cv=5, scoring='recall')
        grid_search.fit(X_train, y_train)

        best_gb = grid_search.best_estimator_
        y_pred_gb = best_gb.predict(X_test)

        results.append({
            'Algorithm': 'Gradient Boosting',
            'Best Hyperparameters': grid_search.best_params_,
            'Accuracy': accuracy_score(y_test, y_pred_gb),
            'Precision': precision_score(y_test, y_pred_gb),
            'Recall': recall_score(y_test, y_pred_gb),
            'F1 Score': f1_score(y_test, y_pred_gb)
        })

        plot_confusion_matrix(y_test, y_pred_gb, "Gradient Boosting")
        plot_feature_importance(best_gb, feature_names, "Gradient Boosting", X_test, y_test)

    def train_gru():
        X_train_gru = np.expand_dims(X_train, axis=1)
        X_test_gru = np.expand_dims(X_test, axis=1)

        model = Sequential([
            GRU(64, input_shape=(X_train_gru.shape[1], X_train_gru.shape[2]), activation='tanh', return_sequences=True),
            Dropout(0.2),
            GRU(32, activation='tanh'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(X_train_gru, y_train, validation_split=0.2, epochs=50, batch_size=32, callbacks=[early_stopping], verbose=0)
        y_pred_gru = (model.predict(X_test_gru) > 0.5).astype(int).flatten()

        results.append({
            'Algorithm': 'GRU',
            'Best Hyperparameters': '64 Hidden Units (Fixed)',
            'Accuracy': accuracy_score(y_test, y_pred_gru),
            'Precision': precision_score(y_test, y_pred_gru),
            'Recall': recall_score(y_test, y_pred_gru),
            'F1 Score': f1_score(y_test, y_pred_gru)
        })

        plot_confusion_matrix(y_test, y_pred_gru, "GRU")

    print("Training KNN...")
    train_knn()

    print("\nTraining Logistic Regression...")
    train_logistic_regression()

    print("\nTraining Gradient Boosting...")
    train_gradient_boosting()

    print("\nTraining GRU...")
    train_gru()

    results_df = pd.DataFrame(results)
    print("\nFinal Model Performance:")
    print(results_df)

if __name__ == "__main__":
    main_training()
