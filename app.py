
from flask import Flask, render_template
from pathlib import Path
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cybersecurity_data.csv"
RESULTS_PATH = BASE_DIR / "outputs" / "detection_results.csv"
METRICS_PATH = BASE_DIR / "outputs" / "evaluation_metrics.csv"
MODEL_PATH = BASE_DIR / "models" / "cybersecurity_model.joblib"

app = Flask(__name__)


@app.route("/")
def dashboard():
    data = pd.read_csv(DATA_PATH)
    results = pd.read_csv(RESULTS_PATH)
    metrics = pd.read_csv(METRICS_PATH)
    model = joblib.load(MODEL_PATH)

    total_records = len(data)
    normal_count = int((data["threat_label"] == 0).sum())
    threat_count = int((data["threat_label"] == 1).sum())

    latest_alerts = results.to_dict(orient="records")

    metric_values = metrics.iloc[0].to_dict()
    feature_importance = sorted(
        [
            {
                "feature": feature,
                "importance": round(float(value), 4)
            }
            for feature, value in zip(
                data.drop(columns=["threat_label"]).columns,
                model.feature_importances_
            )
        ],
        key=lambda item: item["importance"],
        reverse=True
    )

    return render_template(
        "dashboard.html",
        total_records=total_records,
        normal_count=normal_count,
        threat_count=threat_count,
        latest_alerts=latest_alerts,
        metrics=metric_values,
        feature_importance=feature_importance
    )


if __name__ == "__main__":
    app.run(debug=True)
