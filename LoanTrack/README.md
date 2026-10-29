# 🏦 LoanTrack — Loan Eligibility Prediction System

An end-to-end Machine Learning project that predicts loan eligibility using **13 ML models** (including XGBoost, ensemble methods, and a neural network), served through a **beautiful web UI** and **REST API**.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)

---

## 📁 Project Structure

```
LoanTrack/
├── app/                          # Web server & frontend
│   ├── main.py                   # FastAPI REST API
│   ├── schemas.py                # Pydantic request/response schemas
│   └── static/                   # Web interface (HTML, CSS, JS)
├── config/
│   └── settings.py               # Centralized configurations & paths
├── data/
│   └── loan_data.csv             # Raw Kaggle dataset
├── models/                       # Saved trained model artifacts (.pkl)
│   └── xgboost_loan_model.pkl
├── visualizations/               # Generated evaluation charts (.png)
├── src/loan_ml/                  # Core Python ML source code
│   ├── data/loader.py            # Data loading & minority upsampling
│   ├── preprocessing/pipeline.py # Imputation, encoding, scaling pipeline
│   ├── training/                 # ML model training & evaluation routines
│   │   ├── trainer.py            # 13 ML models + ensemble setup
│   │   └── evaluator.py          # Metric scoring & summary tables
│   ├── plotting/                 # Chart generation routines
│   │   └── plots.py              # Bar charts, ROC curves, confusion matrices
│   └── inference/predictor.py    # Prediction pipeline & artifact loader
├── tests/                        # Automated unit & integration tests
├── cli.py                        # Unified CLI runner (train/predict/serve)
├── train.py                      # Training pipeline execution script
├── requirements.txt              # Project Python dependencies
├── setup.py                      # Package installation config
└── Dockerfile                    # Container definition
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd LoanTrack
pip install -r requirements.txt
```

### 2. Train the models

```bash
python train.py
```

This will:
- Download the Kaggle Loan Prediction dataset
- Run preprocessing (imputation, encoding, scaling)
- Train 13 ML models (Logistic Regression, Decision Tree, Random Forest, Extra Trees, SVM, KNN, Naive Bayes, AdaBoost, Gradient Boosting, MLP Neural Net, XGBoost, + 2 Voting Ensembles)
- Generate comparison charts in `visualizations/`
- Save the best model (XGBoost) to `models/`

### 3. Launch the Web UI

```bash
python cli.py serve
```

Open **http://127.0.0.1:8000** in your browser to use the beautiful prediction interface.

### 4. Run tests

```bash
pytest tests/ -v
```

---

## 🖥️ CLI Commands

| Command | Description |
|---------|-------------|
| `python cli.py train` | Train with log transform (default) |
| `python cli.py train --no-log` | Train without log transform |
| `python cli.py train --both` | Run both with & without log |
| `python cli.py predict` | Demo inference on sample applicant |
| `python cli.py serve` | Start FastAPI web server |
| `python cli.py serve --port 5000` | Custom port |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Predict loan eligibility |

### Example API Request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "1",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 2000,
    "LoanAmount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1.0,
    "Property_Area": "Urban"
  }'
```

---

## 🐳 Docker

```bash
docker build -t loantrack .
docker run -p 8000:8000 loantrack
```

---

## 🧠 Models Evaluated

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Extra Trees
5. SVM (RBF kernel)
6. K-Nearest Neighbors
7. Gaussian Naive Bayes
8. AdaBoost
9. Gradient Boosting
10. Dense Neural Network (MLP)
11. **XGBoost** ⭐ (Best Model)
12. Voting Ensemble 1 (all models)
13. Voting Ensemble 2 (top-3 boosters)

---

## 📊 Dataset

- **Source**: [Kaggle Loan Prediction Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- **Records**: 614
- **Features**: 13 (11 input + 1 target + 1 ID)
- **Target**: `Loan_Status` (Y = Approved, N = Rejected)
