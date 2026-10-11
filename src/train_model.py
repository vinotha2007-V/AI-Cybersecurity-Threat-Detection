
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cybersecurity_data.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def train_model():
    print("=" * 55)
    print("AI-POWERED CYBERSECURITY THREAT DETECTION")
    print("=" * 55)

    # 1. Load dataset
    df = pd.read_csv(DATA_PATH)

    print("\nDataset loaded successfully!")
    print("Total records:", len(df))
    print("Dataset shape:", df.shape)

    # 2. Separate features and target
    X = df.drop(columns=["threat_label"])
    y = df["threat_label"]

    print("\nClass distribution:")
    print(y.value_counts().sort_index())

    # 3. Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))

    # 4. Create and train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    print("\nTraining Random Forest model...")
    model.fit(X_train, y_train)

    # 5. Make predictions on unseen test records
    y_pred = model.predict(X_test)

    # 6. Evaluate the model
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, y_pred, zero_division=0
        )
    }

    print("\nMODEL EVALUATION")
    print("-" * 35)

    for name, value in metrics.items():
        print(f"{name.replace('_', ' ').title()}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 7. Save trained model
    model_path = MODEL_DIR / "cybersecurity_model.joblib"
    joblib.dump(model, model_path)

    # 8. Save evaluation metrics
    metrics_path = OUTPUT_DIR / "evaluation_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    print("\nModel saved to:", model_path)
    print("Metrics saved to:", metrics_path)
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    train_model()
