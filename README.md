
# Credit Card Fraud Detection – Production-Grade ML Prediction Service

**A Machine Learning project that detects fraudulent credit card transactions using Python, FastAPI, Docker, DVC, GitHub Actions, Prometheus, and Grafana.**

## Project Overview

Credit card fraud is a major problem in digital banking. This project uses a Machine Learning model to identify potentially fraudulent transactions.

The trained model is deployed through a FastAPI REST API. Docker is used for containerization, DVC for dataset and pipeline versioning, GitHub Actions for CI/CD, and Prometheus for API monitoring.

## Objectives

- Detect fraudulent and genuine credit card transactions.
- Train and evaluate a Machine Learning classification model.
- Build a REST API to serve predictions.
- Track datasets and ML pipelines using DVC.
- Containerize the application using Docker.
- Automate testing through GitHub Actions.
- Monitor API performance using Prometheus and Grafana.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Scikit-learn | Machine Learning |
| Pandas & NumPy | Data processing |
| FastAPI | REST API development |
| Git & GitHub | Version control |
| DVC | Dataset and pipeline versioning |
| Docker | Application containerization |
| GitHub Actions | CI/CD automation |
| Prometheus | Metrics monitoring |
| Grafana | Monitoring visualization |

## Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── app/
│   └── main.py
│
├── data/
│   └── creditcard.csv.dvc
│
├── models/
│   └── model.joblib
│
├── monitoring/
│   ├── prometheus.yml
│   └── grafana-datasource.yml
│
├── reports/
│   └── metrics.json
│
├── src/
│   └── train.py
│
├── tests/
│   └── test_api.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── params.yaml
├── requirements.txt
└── README.md
```

## Dataset

The project uses the Credit Card Fraud Detection dataset containing 284,807 transactions.

- Total transactions: 284,807
- Fraudulent transactions: 492
- Genuine transactions: 284,315

The dataset contains transaction features, including Time, Amount, and anonymized features V1 to V28.

## Model Performance

The trained model achieved the following evaluation results:

| Metric | Result |
|---|---:|
| Precision | 0.0610 |
| Recall | 0.9184 |
| F1 Score | 0.1144 |
| ROC-AUC | 0.9721 |
| Average Precision | 0.7190 |

The model achieves high fraud recall, meaning it identifies a large proportion of fraudulent transactions. Precision is comparatively low, so some genuine transactions may also be flagged.

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Home page |
| `/health` | GET | Check API and model status |
| `/predict` | POST | Predict transaction status |
| `/metrics` | GET | View Prometheus metrics |

## Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/saloniverma204-lgtm/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the API

```bash
uvicorn app.main:app --reload
```

Open the API documentation:

http://127.0.0.1:8000/docs

## Running with Docker

Start the application and monitoring services:

```bash
docker compose up -d --build
```

Services:

- FastAPI: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

Stop the services:

```bash
docker compose down
```

## CI/CD

GitHub
