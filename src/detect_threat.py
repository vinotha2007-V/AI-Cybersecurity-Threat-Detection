
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "cybersecurity_model.joblib"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def detect_threats():
    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Two sample network activities for simulation
    samples = pd.DataFrame([
        {
            "duration": 120,
            "source_bytes": 2000,
            "destination_bytes": 3000,
            "failed_logins": 0,
            "connection_count": 10,
            "port_risk": 0
        },
        {
            "duration": 50,
            "source_bytes": 95000,
            "destination_bytes": 8000,
            "failed_logins": 8,
            "connection_count": 90,
            "port_risk": 1
        }
    ])

    sample_names = [
        "Normal Activity Simulation",
        "Suspicious Activity Simulation"
    ]

    # Predict using the trained model
    predictions = model.predict(samples)

    results = []

    print("=" * 55)
    print("AI CYBERSECURITY THREAT DETECTION")
    print("=" * 55)

    for name, features, prediction in zip(
        sample_names,
        samples.to_dict(orient="records"),
        predictions
    ):
        is_threat = int(prediction) == 1

        status = "THREAT DETECTED" if is_threat else "NORMAL"
        severity = "HIGH" if is_threat else "INFO"
        alert = (
            "Investigate suspicious activity"
            if is_threat
            else "No alert generated"
        )

        print(f"\nActivity: {name}")
        print(f"Prediction: {status}")
        print(f"Alert severity: {severity}")
        print(f"Action: {alert}")

        results.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "activity": name,
            **features,
            "predicted_label": int(prediction),
            "status": status,
            "severity": severity,
            "alert": alert
        })

    # Save detection results
    results_path = OUTPUT_DIR / "detection_results.csv"
    pd.DataFrame(results).to_csv(results_path, index=False)

    print("\nDetection simulation completed.")
    print("Results saved to:", results_path)


if __name__ == "__main__":
    detect_threats()
