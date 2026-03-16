# NIDS-ML: Machine Learning Network Intrusion Detection System

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-GPL--3.0-blue)

A real-time Network Intrusion Detection System that uses machine learning to classify network traffic into 15 attack types. Built as a Final Year Project, it combines a full ML pipeline (data exploration through explainability), a FastAPI prediction service, a live packet-capture engine, and monitoring dashboards.

---

## Architecture

```
                          +---------------------+
                          |   CIC-IDS2017 Data  |
                          |   (2M+ flows)       |
                          +---------+-----------+
                                    |
                     Notebooks 01-03 (EDA, Preprocessing,
                                    Feature Engineering)
                                    |
                                    v
                          +---------------------+
                          |  Model Training     |
                          |  RF + XGBoost       |
                          |  (Notebook 04)      |
                          +---------+-----------+
                                    |
                          +---------+-----------+
                          |  SHAP / LIME        |
                          |  Explainability     |
                          |  (Notebook 05)      |
                          +---------+-----------+
                                    |
                   Saved models (.pkl) + scaler + encoder
                                    |
              +---------------------+---------------------+
              |                                           |
              v                                           v
   +--------------------+                    +------------------------+
   |  FastAPI Service   |<--- REST ---------+|  Flask Dashboard       |
   |  (port 8000)       |                    |  (port 5001)           |
   |  /predict           |                    |  Real-time status      |
   |  /batch_predict     |                    |  Attack distribution   |
   |  /stats             |                    +------------------------+
   |  /metrics (Prom)    |
   +---------+----------+
             |
             |  Prometheus scrape (/metrics)
             v
   +--------------------+         +------------------------+
   |  Prometheus        |-------->|  Grafana               |
   |  (port 9090)       |         |  (port 3000)           |
   |  30-day retention  |         |  11 dashboard panels   |
   +--------------------+         +------------------------+
             ^
             |
   +--------------------+
   |  NIDS Engine       |
   |  (Real-time)       |
   |  Scapy capture --> |--- flow aggregation --> feature extraction --> API predict
   +---------+----------+
             |
             |  Attack detected?
             v
   +--------------------+
   |  Alerting          |
   |  Email (SMTP)      |
   |  Slack (Webhook)   |
   +--------------------+
```

## Features

- **ML Classification** — Random Forest + XGBoost ensemble detecting 15 attack types with 99.89% accuracy
- **Real-Time Detection** — Live packet capture via Scapy, 5-tuple flow aggregation, and 84-feature extraction
- **Explainability** — SHAP and LIME explanations for every prediction (top 5 contributing features)
- **REST API** — FastAPI with Swagger docs, batch prediction, rate limiting, and optional API key auth
- **Alerting** — Email (Gmail/Outlook SMTP) and Slack webhook notifications with configurable severity thresholds and cooldown periods
- **Session Logging** — Timestamped session directories with separate engine, prediction, attack, and summary logs per run
- **One-Command Launch** — `run_nids.py` starts both the API server and NIDS engine together
- **Monitoring** — Prometheus metrics + pre-built Grafana dashboard (latency, attack trends, model health)
- **Flask Dashboard** — Lightweight real-time status page with auto-refresh and attack distribution charts
- **Benchmarking & Evaluation** — Performance benchmarks (latency/throughput) and randomised accuracy testing with confidence intervals
- **Docker Support** — Single `docker-compose up` deploys the full stack (API + Prometheus + Grafana)

## Tech Stack

| Layer | Technology |
|---|---|
| ML Models | scikit-learn (Random Forest), XGBoost |
| Explainability | SHAP, LIME |
| Data Processing | NumPy, Pandas |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Flask, Chart.js |
| Packet Capture | Scapy |
| Monitoring | Prometheus, Grafana |
| Containerisation | Docker, Docker Compose |
| Security | slowapi (rate limiting), API key auth |

## Project Structure

```
FYP-NIDS-ML/
├── api/                            # FastAPI prediction service
│   ├── app.py                      # Main API (endpoints, model loading, SHAP)
│   ├── auth.py                     # API key authentication
│   ├── rate_limit.py               # Rate limiting middleware
│   ├── test_client.py              # API test suite
│   └── requirements.txt
│
├── src/                            # Real-time NIDS engine
│   ├── nids_engine.py              # Main orchestrator (session logging)
│   ├── packet_capture.py           # Scapy packet capture
│   ├── flow_aggregator.py          # 5-tuple flow aggregation
│   ├── feature_extractor.py        # 84-feature extraction
│   ├── alerting.py                 # Email + Slack alert management
│   ├── constants.py                # Shared feature name definitions
│   └── requirements.txt
│
├── testing/                        # Kali Linux attack testing suite
│   ├── vulnerable_webapp.py        # Intentionally vulnerable Flask app
│   ├── setup_services.py           # FTP/SSH test service setup
│   ├── run_kali_tests.py           # Automated Kali attack runner
│   ├── parse_kali_results.py       # Results parser
│   └── verify_interface.py         # Network interface checker
│
├── dashboard/                      # Flask monitoring UI
│   ├── app.py                      # Dashboard routes
│   ├── templates/index.html        # Web interface
│   └── static/style.css            # Dark theme (Catppuccin Mocha)
│
├── monitoring/                     # Prometheus + Grafana configs
│   ├── prometheus/prometheus.yml
│   └── grafana/
│       ├── provisioning/           # Auto-configure datasource
│       └── dashboards/             # Pre-built NIDS dashboard
│
├── figures/                        # Generated visualisations
│   ├── shap/                       # SHAP explanations
│   └── lime/                       # LIME explanations
│
├── results/                        # Evaluation & benchmark results (CSV/TXT)
├── config/nids_config.yaml         # NIDS engine configuration
│
├── 01_data_exploration.ipynb       # EDA & class distributions
├── 02_data_preprocessing.ipynb     # Cleaning & encoding
├── 03_feature_engineering.ipynb    # Feature scaling & selection
├── 04_model_training.ipynb         # RF + XGBoost training
├── 05_xai_explainability.ipynb     # SHAP + LIME analysis
├── 06_api_deployment.ipynb         # Deployment walkthrough
├── 04_model_training_improved.py   # Standalone training script
│
├── run_nids.py                     # One-command launcher (API + engine)
├── benchmark_suite.py              # Performance benchmarks
├── randomised_evaluation.py        # Randomised accuracy evaluation
├── test_attack_detection.py        # Attack detection validation
├── SYSTEM_EXPLANATION.md           # System architecture documentation
│
├── Dockerfile                      # Multi-stage API container
├── docker-compose.yml              # Full stack orchestration
└── .dockerignore
```

## Quick Start

### Prerequisites

- Python 3.11+
- [Npcap](https://npcap.com/) (Windows) or libpcap (Linux/Mac) — for live packet capture
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) — optional, for the monitoring stack

### 1. Clone & Set Up

```bash
git clone https://github.com/C00286234/FYP-NIDS-ML.git
cd FYP-NIDS-ML

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r api/requirements.txt
pip install -r src/requirements.txt
pip install -r dashboard/requirements.txt
```

### 2. Generate Models

Models are not included in the repo due to size (~147MB). Generate them by running the training notebook:

```bash
# Install Jupyter if needed
pip install jupyter

# Option A: Run the notebook interactively
jupyter notebook 04_model_training.ipynb

# Option B: Run the standalone training script
python 04_model_training_improved.py
```

> **Note:** Training requires the CIC-IDS2017 dataset in `data/processed/cicids2017_cleaned.csv`. Download the raw dataset from [UNB CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) and run notebooks 01-03 to preprocess it. Alternatively, run notebooks 01 through 04 in sequence.

This creates the following in `models/`:
| File | Purpose |
|---|---|
| `random_forest_model.pkl` | Primary model (~140MB) |
| `xgboost_model.pkl` | Fallback model (~7MB) |
| `scaler.pkl` | StandardScaler for 84 features |
| `feature_selector.pkl` | SelectKBest (84 → 50 features) |
| `label_encoder.pkl` | Maps 15 classes to numeric labels |
| `training_config.pkl` | Training metadata |

### 3. Run the API

```bash
cd api
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Verify: open [http://localhost:8000/docs](http://localhost:8000/docs) for Swagger UI.

### 4. Run the Dashboard (optional)

```bash
cd dashboard
python app.py
```

Open [http://localhost:5001](http://localhost:5001).

### 5. Run the NIDS Engine (optional, requires admin)

```bash
# Requires administrator/root privileges for packet capture

# Option A: Launch both API + engine together
python run_nids.py

# Option B: Specify a network interface
python run_nids.py -i Ethernet

# Option C: Start only the API or only the engine
python run_nids.py --api-only
python run_nids.py --engine-only    # API must already be running
```

Session logs are saved to `logs/sessions/YYYYMMDD_HHMMSS/` with separate engine, prediction, attack, and summary files.

## Docker Deployment

Deploy the API with full monitoring in one command:

```bash
docker-compose up -d
```

| Service | URL | Credentials |
|---|---|---|
| NIDS API | [http://localhost:8000](http://localhost:8000) | — |
| Swagger Docs | [http://localhost:8000/docs](http://localhost:8000/docs) | — |
| Prometheus | [http://localhost:9090](http://localhost:9090) | — |
| Grafana | [http://localhost:3000](http://localhost:3000) | admin / nids-admin |

```bash
# View logs
docker-compose logs -f nids-api

# Stop
docker-compose down

# Rebuild after code changes
docker-compose up -d --build
```

## API Reference

| Method | Endpoint | Description | Rate Limit |
|---|---|---|---|
| `GET` | `/` | Welcome message + endpoint list | — |
| `GET` | `/health` | Health check (model status, uptime) | — |
| `GET` | `/stats` | Usage statistics (predictions, attack distribution) | — |
| `GET` | `/metrics` | Prometheus metrics (text format) | — |
| `GET` | `/features` | List of 84 required feature names | — |
| `GET` | `/classes` | List of 15 attack classes | — |
| `POST` | `/predict` | Single flow prediction + SHAP explanation | 60/min |
| `POST` | `/batch_predict` | Batch prediction (up to 1000 flows) | 30/min |
| `GET` | `/docs` | Interactive Swagger documentation | — |

### Example: Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"Src Port": 443, "Dst Port": 80, "Protocol": 6, ...}}'
```

Response:
```json
{
  "prediction": "DDoS",
  "confidence": 0.9847,
  "is_attack": true,
  "probabilities": {"BENIGN": 0.012, "DDoS": 0.985, ...},
  "explanation": [
    {"feature": "Flow Packets/s", "value": 15234.5, "contribution": 0.42},
    {"feature": "Dst Port", "value": 80.0, "contribution": 0.28}
  ]
}
```

## Monitoring

### Flask Dashboard (port 5001)
Real-time status page showing current attack counts, system health, and attack distribution. Auto-refreshes every 5 seconds.

### Grafana (port 3000)
Pre-built dashboard with 11 panels:
- **Overview** — Total predictions, attacks detected, attack rate %, avg latency
- **Traffic Over Time** — Predictions and attacks as time-series graphs
- **Performance** — Latency percentiles (p50/p95/p99), API request rate
- **Breakdown** — Attack type distribution (pie chart), model status, top attack types

### Prometheus (port 9090)
Scrapes the API every 15 seconds. Key metrics:
- `nids_predictions_total` — Total predictions (by class)
- `nids_attacks_detected_total` — Attacks detected (by type)
- `nids_prediction_latency_seconds` — Prediction latency histogram
- `nids_models_loaded` — Model load status gauge

## ML Pipeline

The ML pipeline is implemented across 6 Jupyter notebooks:

| # | Notebook | Purpose |
|---|---|---|
| 01 | `01_data_exploration.ipynb` | EDA, class distributions, feature correlations |
| 02 | `02_data_preprocessing.ipynb` | Cleaning, missing values, outlier removal |
| 03 | `03_feature_engineering.ipynb` | Feature scaling (StandardScaler), selection (SelectKBest: 84 → 50) |
| 04 | `04_model_training.ipynb` | Random Forest + XGBoost, SMOTE for class imbalance, 5-fold CV |
| 05 | `05_xai_explainability.ipynb` | SHAP (global + local) and LIME explanations |
| 06 | `06_api_deployment.ipynb` | FastAPI deployment walkthrough |

### Dataset

**CIC-IDS2017** from the Canadian Institute for Cybersecurity — 2,099,976 network flows captured over 5 days with 15 attack types injected into normal traffic.

### To Retrain

```bash
# Run the standalone training script
python 04_model_training_improved.py

# Or run notebooks 01-04 in sequence for the full pipeline
jupyter notebook
```

## Model Performance

Evaluated on a held-out test set (311,746 flows):

| Metric | XGBoost | Random Forest |
|---|---|---|
| **Accuracy** | 99.89% | 99.40% |
| **F1 (macro)** | 0.9670 | 0.9640 |
| **Precision** | 97.39% | 99.41% |
| **Recall** | 96.15% | 99.40% |

### Confusion Matrices

<p align="center">
  <img src="figures/xgboost_confusion_matrix_v2.png" alt="XGBoost Confusion Matrix" width="45%">
  <img src="figures/rf_confusion_matrix.png" alt="Random Forest Confusion Matrix" width="45%">
</p>

### Feature Importance (SHAP)

<p align="center">
  <img src="figures/shap/global_importance.png" alt="SHAP Global Feature Importance" width="70%">
</p>

### Known Limitations

The model underperforms on two classes due to limited training samples in CIC-IDS2017:

| Class | F1 Score | Test Samples |
|---|---|---|
| Web Attack - Brute Force | 0.55 | 205 |
| Web Attack - XSS | 0.46 | 101 |

All other classes achieve F1 scores above 0.92.

## Testing & Evaluation

### Benchmark Suite

Measures system performance under load:

```bash
python benchmark_suite.py
```

Reports latency percentiles (p50/p95/p99), single-prediction vs batch throughput, and concurrent request handling. Results are saved to `results/`.

### Randomised Evaluation

Runs attack detection tests with different random seeds and larger samples to produce statistically robust results:

```bash
python randomised_evaluation.py
```

Outputs mean accuracy with standard deviation and 95% confidence intervals across multiple runs.

### Attack Detection Validation

Tests the API against known attack samples from the CIC-IDS2017 dataset:

```bash
python test_attack_detection.py
```

Reports per-class accuracy, precision, recall, and overall detection rates.

### Kali Linux Live Testing

The `testing/` directory contains tools for generating real attack traffic against the NIDS using Kali Linux:

- `vulnerable_webapp.py` — Intentionally vulnerable Flask app (SQL injection, XSS, brute-force targets)
- `setup_services.py` — Sets up FTP/SSH services for brute-force testing
- `run_kali_tests.py` — Automated attack runner using Hydra, Nmap, and Slowloris
- `parse_kali_results.py` — Parses Kali tool output into structured results

## Alerting

The NIDS engine sends real-time alerts when attacks are detected, configurable via `config/nids_config.yaml`.

### Email (SMTP)

Supports Gmail, Outlook, and custom SMTP servers with HTML-formatted alert emails.

```yaml
alerting:
  email:
    enabled: true
    smtp_server: smtp.gmail.com
    smtp_port: 587
    sender: nids-alerts@example.com
    password: ${NIDS_EMAIL_PASSWORD}
    recipients:
      - security-team@example.com
    min_confidence: 0.9
    cooldown_minutes: 5
    severity_threshold: medium
```

### Slack (Webhook)

Posts attack alerts to a Slack channel via incoming webhook.

```yaml
alerting:
  slack:
    enabled: true
    webhook_url: ${NIDS_SLACK_WEBHOOK}
    channel: "#nids-alerts"
    min_confidence: 0.8
    cooldown_minutes: 2
    severity_threshold: medium
```

Alerts include attack type, severity level, confidence score, source/destination IPs, and top contributing features. Cooldown periods prevent alert flooding during sustained attacks.

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NIDS_LOG_LEVEL` | `INFO` | Logging level |
| `NIDS_REQUIRE_AUTH` | `false` | Enable API key authentication |
| `NIDS_API_KEY` | — | API key (if auth enabled) |
| `NIDS_RATE_LIMIT` | `100/minute` | Global rate limit |
| `NIDS_DEBUG` | `false` | Enable hot-reload (dev only) |
| `NIDS_CORS_ORIGINS` | `http://localhost:5001,http://localhost:3000` | Allowed CORS origins |
| `NIDS_API_URL` | `http://localhost:8000` | API URL (for dashboard) |
| `DASHBOARD_PORT` | `5001` | Flask dashboard port |
| `REFRESH_INTERVAL` | `5` | Dashboard refresh rate (seconds) |
| `FLASK_SECRET_KEY` | auto-generated | Flask session secret |
| `NIDS_EMAIL_PASSWORD` | — | SMTP password for email alerts |
| `NIDS_SLACK_WEBHOOK` | — | Slack incoming webhook URL for alerts |

### NIDS Engine Config

The real-time engine is configured via `config/nids_config.yaml`:
- Network interface (default: WiFi, auto-detected)
- Flow timeout (10s) and max flow limits
- API connection details and retry logic
- Email + Slack alerting with severity thresholds and cooldown periods
- Session-based logging to `logs/sessions/`
- Performance tuning (batch size, cleanup interval 5s, stats interval 10s)

## Acknowledgements

- **Dataset**: [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) — Canadian Institute for Cybersecurity, University of New Brunswick
- **Citation**: Sharafaldin, I., Lashkari, A.H., & Ghorbani, A.A. (2018). Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization. *ICISSP 2018*.

## License

This project is licensed under the GNU General Public License v3.0 — see the [LICENSE](LICENSE) file for details.
