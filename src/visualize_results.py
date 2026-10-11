
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cybersecurity_data.csv"
RESULTS_PATH = BASE_DIR / "outputs" / "detection_results.csv"
MODEL_PATH = BASE_DIR / "models" / "cybersecurity_model.joblib"
METRICS_PATH = BASE_DIR / "outputs" / "evaluation_metrics.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def visualize_results():
    data = pd.read_csv(DATA_PATH)
    results = pd.read_csv(RESULTS_PATH)
    metrics = pd.read_csv(METRICS_PATH)
    model = joblib.load(MODEL_PATH)

    # 1. Dataset class distribution
    counts = data["threat_label"].map({
        0: "Normal",
        1: "Threat"
    }).value_counts()

    plt.figure(figsize=(7, 5))
    counts.plot(kind="bar")
    plt.title("Cybersecurity Dataset Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=150)
    plt.close()

    # 2. Detection simulation results
    prediction_counts = results["status"].value_counts()

    plt.figure(figsize=(7, 5))
    prediction_counts.plot(kind="bar")
    plt.title("Threat Detection Simulation Results")
    plt.xlabel("Prediction")
    plt.ylabel("Number of Activities")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "detection_results.png", dpi=150)
    plt.close()

    # 3. Model feature importance
    importance = pd.Series(
        model.feature_importances_,
        index=data.drop(columns=["threat_label"]).columns
    ).sort_values()

    plt.figure(figsize=(8, 5))
    importance.plot(kind="barh")
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.close()

    # 4. Evaluation metrics chart
    metric_values = metrics.iloc[0]

    plt.figure(figsize=(7, 5))
    metric_values.plot(kind="bar")
    plt.title("Model Evaluation Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "evaluation_metrics.png", dpi=150)
    plt.close()

    print("Visualizations generated successfully!")
    print("Saved graphs:")
    print("- outputs/class_distribution.png")
    print("- outputs/detection_results.png")
    print("- outputs/feature_importance.png")
    print("- outputs/evaluation_metrics.png")


if __name__ == "__main__":
    visualize_results()
