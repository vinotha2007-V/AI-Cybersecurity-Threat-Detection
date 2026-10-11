# AI-Powered Cybersecurity Threat Detection System

## Project Overview

The AI-Powered Cybersecurity Threat Detection System is a machine learning project designed to classify simulated cybersecurity activity as normal or potentially suspicious.

It uses Python, Pandas, NumPy, and Scikit-learn for data processing and machine learning, along with Flask and Chart.js to display results through a web dashboard.

## Key Features

- Synthetic cybersecurity dataset generation
- Data analysis and preprocessing
- Random Forest classification model
- Threat detection on simulated records
- Model evaluation using Accuracy, Precision, Recall, and F1 Score
- Feature importance analysis
- Data visualization with Matplotlib
- Interactive Flask dashboard
- Detection results displayed in a table

## Technology Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Model Storage:** Joblib
- **Backend:** Flask
- **Frontend:** HTML, CSS, JavaScript
- **Charts:** Chart.js
- **Visualization:** Matplotlib

## Project Structure

```text
AI-Cybersecurity-Threat-Detection/
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── data/
│   └── cybersecurity_data.csv
├── models/
│   └── cybersecurity_model.joblib
├── outputs/
│   ├── evaluation_metrics.csv
│   ├── detection_results.csv
│   ├── class_distribution.png
│   ├── detection_results.png
│   ├── feature_importance.png
│   └── evaluation_metrics.png
├── src/
│   ├── generate_data.py
│   ├── train_model.py
│   ├── detect_threat.py
│   └── visualize_results.py
└── templates/
    └── dashboard.html
```

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd AI-Cybersecurity-Threat-Detection
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows CMD:

```cmd
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Project

Run these commands from the project root directory.

### Step 1: Generate the dataset

```bash
python src/generate_data.py
```

### Step 2: Train the model

```bash
python src/train_model.py
```

### Step 3: Run threat detection

```bash
python src/detect_threat.py
```

### Step 4: Generate visualizations

```bash
python src/visualize_results.py
```

### Step 5: Start the dashboard

```bash
python app.py
```

Open the following URL in your browser:

http://127.0.0.1:5000

## Model Evaluation

The project evaluates its classification model using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Evaluation results are saved in `outputs/evaluation_metrics.csv`.

**Important:** The current dataset is synthetically generated, and its threat labels are based on predefined rules. High evaluation scores on this dataset do not demonstrate real-world detection performance. Testing with independent, realistic datasets is required before drawing conclusions about practical cybersecurity effectiveness.

## Learning Outcomes

Through this project, I explored:

- Python-based data analysis
- Machine learning model development
- Classification and evaluation metrics
- Cybersecurity threat classification concepts
- Data visualization
- Flask web application development
- Integration of a machine learning model with a dashboard

## Future Enhancements

- Evaluate on public cybersecurity datasets
- Add more robust data preprocessing
- Compare multiple machine learning algorithms
- Improve false-positive and false-negative analysis
- Add dashboard filters and historical analytics
- Develop a real-time log ingestion pipeline

## Disclaimer

This project is intended for educational purposes and uses simulated cybersecurity data. It is not a production-ready intrusion detection or security monitoring system.

## Author

Developed as a student project to explore machine learning, data analysis, and cybersecurity.