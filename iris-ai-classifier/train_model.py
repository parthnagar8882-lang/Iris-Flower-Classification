"""Train and evaluate a KNN classifier for the UCI Iris dataset."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "iris.data"
MODEL_DIR = BASE_DIR / "model"
FEATURE_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
COLUMN_NAMES = FEATURE_COLUMNS + ["species"]


def load_dataset():
    """Load the headerless Iris data file and remove completely blank rows."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset file was not found at: {DATA_PATH}\n"
            "Place iris.data inside the data folder before training."
        )

    dataset = pd.read_csv(DATA_PATH, header=None, names=COLUMN_NAMES)
    dataset = dataset.dropna(how="all")
    dataset = dataset.dropna().reset_index(drop=True)
    return dataset


def print_dataset_analysis(dataset):
    """Print a small, beginner-friendly summary of the dataset."""
    print("\n" + "=" * 60)
    print("IRIS DATASET ANALYSIS")
    print("=" * 60)
    print(f"\nDataset shape: {dataset.shape}")
    print("\nFirst 5 rows:")
    print(dataset.head())
    print("\nDataset information:")
    dataset.info()
    print("\nMissing values:")
    print(dataset.isnull().sum())
    print("\nClass distribution:")
    print(dataset["species"].value_counts())
    print("\nDescriptive statistics:")
    print(dataset.describe())


def main():
    dataset = load_dataset()
    print_dataset_analysis(dataset)

    X = dataset[FEATURE_COLUMNS]
    y = dataset["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # The scaler is fitted only on training data to prevent data leakage.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    best_k = 1
    best_accuracy = -1.0
    print("\n" + "=" * 60)
    print("TESTING K VALUES")
    print("=" * 60)
    for k_value in range(1, 16):
        candidate_model = KNeighborsClassifier(n_neighbors=k_value)
        candidate_model.fit(X_train_scaled, y_train)
        candidate_accuracy = accuracy_score(y_test, candidate_model.predict(X_test_scaled))
        print(f"K = {k_value}: Test Accuracy = {candidate_accuracy:.4f}")
        # Strictly greater preserves the first (smallest) K in a tie.
        if candidate_accuracy > best_accuracy:
            best_accuracy = candidate_accuracy
            best_k = k_value

    print(f"\nSelected Best K Value: {best_k}")
    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
    recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

    print("\n" + "=" * 60)
    print("FINAL MODEL EVALUATION")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "knn_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    metrics = {
        "model_name": "K-Nearest Neighbors (KNN)",
        "best_k": int(best_k),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "total_samples": int(len(dataset)),
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
        "total_features": int(len(FEATURE_COLUMNS)),
        "total_classes": int(y.nunique()),
    }
    with (MODEL_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nSaved model files:")
    print(MODEL_DIR / "knn_model.pkl")
    print(MODEL_DIR / "scaler.pkl")
    print(MODEL_DIR / "metrics.json")


if __name__ == "__main__":
    main()
