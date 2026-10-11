
import pandas as pd
import numpy as np
from pathlib import Path


def generate_dataset(rows=5000):
    np.random.seed(42)

    data = {
        "duration": np.random.randint(1, 3000, rows),
        "source_bytes": np.random.randint(0, 100000, rows),
        "destination_bytes": np.random.randint(0, 100000, rows),
        "failed_logins": np.random.randint(0, 10, rows),
        "connection_count": np.random.randint(1, 100, rows),
        "port_risk": np.random.randint(0, 2, rows),
    }

    df = pd.DataFrame(data)

    risk_score = (
        (df["failed_logins"] >= 5).astype(int)
        + (df["connection_count"] >= 70).astype(int)
        + df["port_risk"]
        + (df["source_bytes"] >= 80000).astype(int)
    )

    df["threat_label"] = (risk_score >= 2).astype(int)

    output_path = Path(__file__).resolve().parent.parent / "data"
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / "cybersecurity_data.csv"
    df.to_csv(file_path, index=False)

    print("Synthetic cybersecurity dataset created!")
    print(f"Total records: {len(df)}")
    print(f"Normal records: {(df['threat_label'] == 0).sum()}")
    print(f"Threat records: {(df['threat_label'] == 1).sum()}")
    print(f"Saved to: {file_path}")


if __name__ == "__main__":
    generate_dataset()
