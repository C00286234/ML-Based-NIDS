# NIDS-ML: Machine Learning Network Intrusion Detection System

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-GPL--3.0-blue)

A real-time Network Intrusion Detection System that uses machine learning to classify network traffic into 16 classes: `BENIGN` plus 15 attack types. The repo combines a script-based ML pipeline, a FastAPI prediction service, a live packet-capture engine, alerting, monitoring, and lab-style attack testing.

## Architecture

```text
Raw CIC-IDS2017 day CSVs
    -> 02_data_preprocessing.py
    -> 03_feature_engineering.py
    -> 04_model_training_improved.py
    -> models/*.pkl + data/processed/selected_features.txt
    -> FastAPI API (port 8000)
         -> /predict
         -> /batch_predict
         -> /stats
         -> /metrics
         -> /security
    -> Flask dashboard (port 5001)
    -> Prometheus (port 9090)
    -> Grafana (port 3000)

Live traffic
    -> src/packet_capture.py
    -> src/flow_aggregator.py
    -> src/feature_extractor.py
    -> API prediction
    -> src/alerting.py
    -> logs/sessions/YYYYMMDD_HHMMSS/
```

## Features

- Soft-voting Random Forest + XGBoost inference across 16 classes
- Real-time packet capture with Scapy and bidirectional 5-tuple flow aggregation
- 84 CIC-IDS-style flow features with shared schema in `src/constants.py`
- SHAP top-feature explanations returned by `/predict`
- Batch inference endpoint for higher throughput
- Optional API key auth and per-endpoint rate limiting
- Email and Slack alerting with thresholds, severity filters, and cooldowns
- Session logs plus rolling logs under `logs/`
- Flask dashboard plus Prometheus/Grafana monitoring
- Benchmark, sampled evaluation, demo replay, and Kali attack-testing helpers

## Project Structure

```text
ML-BASED-NIDS/
|-- api/                          # FastAPI service, auth, rate limiting, test client
|-- src/                          # Packet capture engine, flows, features, alerting
|-- dashboard/                    # Flask status dashboard
|-- monitoring/                   # Prometheus and Grafana config
|-- testing/                      # Vulnerable lab app and Kali test helpers
|-- config/                       # Engine configuration
|-- data/processed/               # Cleaned CSV, arrays, selected feature list
|-- figures/                      # Saved SHAP/LIME and model plots
|-- logs/                         # Rolling logs plus session folders
|-- models/                       # Saved models and preprocessing artefacts
|-- results/                      # Benchmark and evaluation reports
|-- 02_data_preprocessing.py
|-- 03_feature_engineering.py
|-- 04_model_training_improved.py
|-- benchmark_suite.py
|-- demo_traffic_simulator.py
|-- randomised_evaluation.py
|-- run_nids.py
|-- test_attack_detection.py
|-- test_capture.py
|-- Dockerfile
|-- docker-compose.yml
`-- README.md
```

## Quick Start

### Prerequisites

- Python 3.11+
- [Npcap](https://npcap.com/) on Windows or libpcap on Linux/macOS for live capture
- Docker Desktop if you want the Prometheus/Grafana stack

### 1. Clone and install dependencies

```bash
git clone https://github.com/C00286234/ML-Based-NIDS.git
cd ML-BASED-NIDS

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r api/requirements.txt
pip install -r src/requirements.txt
pip install -r dashboard/requirements.txt
```

### 2. Use the included artefacts or retrain

This checkout already contains model artefacts in `models/` and processed data in `data/processed/`, so you can start the API immediately.

To regenerate everything from raw CIC-IDS2017 day files, place:

```text
~/nids_extras/CICIDS2017_improved/
  monday.csv
  tuesday.csv
  wednesday.csv
  thursday.csv
  friday.csv
```

Then run:

```bash
python 02_data_preprocessing.py
python 03_feature_engineering.py
python 04_model_training_improved.py
```

If `data/processed/cicids2017_cleaned.csv` already exists, you can skip step 02.

Main artefacts written to `models/`:

| File | Purpose |
|---|---|
| `random_forest_model.pkl` | Random Forest classifier |
| `xgboost_model.pkl` | XGBoost classifier |
| `scaler.pkl` | `StandardScaler` for the 84 raw features |
| `feature_selector.pkl` | Feature selector used before inference |
| `label_encoder.pkl` | Maps 16 classes to numeric labels |
| `training_config.pkl` | Saved training metadata |

### 3. Run the API

```bash
cd api
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Run the dashboard

```bash
cd dashboard
python app.py
```

Dashboard: [http://localhost:5001](http://localhost:5001)

### 5. Run the live NIDS engine

Packet capture usually requires administrator or root privileges.

```bash
# Start API + engine together
python run_nids.py

# Pick a specific interface
python run_nids.py -i Ethernet

# Start only one side
python run_nids.py --api-only
python run_nids.py --engine-only
```

Session logs are written under `logs/sessions/YYYYMMDD_HHMMSS/`. Rolling logs also live under `logs/api/` and `logs/nids/`.

## Docker Deployment

The included compose stack is configured for the API, Prometheus, and Grafana:

```bash
docker-compose up -d
```

| Service | URL | Credentials |
|---|---|---|
| API | [http://localhost:8000](http://localhost:8000) | none |
| Swagger | [http://localhost:8000/docs](http://localhost:8000/docs) | none |
| Prometheus | [http://localhost:9090](http://localhost:9090) | none |
| Grafana | [http://localhost:3000](http://localhost:3000) | `admin` / `nids-admin` |

The compose file does not start the live packet-capture engine or the Flask dashboard.

Useful commands:

```bash
docker-compose logs -f nids-api
docker-compose down
docker-compose up -d --build
```

## API Reference

| Method | Endpoint | Description | Rate limit |
|---|---|---|---|
| `GET` | `/` | Welcome payload and endpoint list | none |
| `GET` | `/health` | Health check and model-load status | none |
| `GET` | `/stats` | Runtime metrics and attack distribution | none |
| `GET` | `/metrics` | Prometheus metrics | none |
| `GET` | `/features` | 84 expected feature names and selected-feature list | none |
| `GET` | `/classes` | 16 class labels (`BENIGN` + 15 attacks) | none |
| `GET` | `/security` | Auth, rate-limit, and metrics configuration | none |
| `POST` | `/predict` | Single-flow prediction with SHAP top features | `60/minute` |
| `POST` | `/batch_predict` | Batch prediction for up to 1000 flows | `30/minute` |

### Request shapes

Examples below are abbreviated for readability. Use `/features` for the full 84-feature contract and ordering.

- `/predict` expects a dictionary of all 84 named features:

```json
{
  "features": {
    "Src Port": 443,
    "Dst Port": 80,
    "Protocol": 6
  },
  "flow_id": "optional-flow-id"
}
```

- `/batch_predict` expects each flow as an ordered list of 84 numeric values in the order returned by `/features`:

```json
{
  "flows": [
    {
      "features": [443, 80, 6, 1000000, 10, 12]
    }
  ]
}
```

### Example single prediction response

```json
{
  "prediction": "DDoS",
  "prediction_id": 2,
  "confidence": 0.9847,
  "probabilities": {
    "BENIGN": 0.012,
    "DDoS": 0.985
  },
  "top_features": [
    {
      "feature": "Flow Packets/s",
      "value": 15234.5,
      "contribution": 0.42
    },
    {
      "feature": "Dst Port",
      "value": 80.0,
      "contribution": 0.28
    }
  ],
  "is_attack": true,
  "model_agreement": true,
  "timestamp": "2026-03-10T17:41:13.123456"
}
```

`/batch_predict` returns per-flow predictions and confidence, but not SHAP explanations.

## Monitoring

### Flask dashboard

The Flask dashboard polls the API and shows:

- health and connectivity
- total predictions and attacks
- attack-rate percentage
- attack distribution data for charts
- auth and rate-limit status

### Prometheus and Grafana

Prometheus scrapes `/metrics`, and Grafana ships with a pre-built dashboard. Key metrics include:

- `nids_predictions_total`
- `nids_attacks_detected_total`
- `nids_prediction_latency_seconds`
- `nids_models_loaded`

## ML Pipeline

This checkout is script-based rather than notebook-based:

| Step | Script | Purpose |
|---|---|---|
| 02 | `02_data_preprocessing.py` | Combine the five day files, consolidate labels, clean inf/NaN values, and save the cleaned CSV |
| 03 | `03_feature_engineering.py` | Encode labels, create a stratified 70/15/15 split, scale features, select the top 50, apply SMOTE, and save arrays plus preprocessing artefacts |
| 04 | `04_model_training_improved.py` | Train and evaluate Random Forest and XGBoost models, optionally tune hyperparameters, and save model artefacts |

Saved explainability plots and prior experiment outputs remain in `figures/` and `results/`.

### Dataset

The project uses CIC-IDS2017 from the Canadian Institute for Cybersecurity. In this repo the label space is 16 classes total: `BENIGN` plus 15 attack types.

## Model Performance

The runtime API averages Random Forest and XGBoost probabilities for inference. The saved file `results/model_results_v2.csv` currently reports these individual model metrics:

| Model | Split | Accuracy | F1 (macro) | Precision (macro) | Recall (macro) |
|---|---|---|---|---|---|
| Random Forest | Validation | 99.40% | 0.8732 | 0.8753 | 0.8771 |
| XGBoost | Validation | 99.45% | 0.9034 | 0.9318 | 0.8990 |
| XGBoost | Test | 99.44% | 0.8337 | 0.8780 | 0.8123 |

### Saved operational benchmarks

Included reports in `results/` currently show:

- `benchmark_20260310_173613.txt`: average single `/predict` latency 2262.9 ms, about 22.5 ms/flow at batch size 100, and about 98.3x speedup versus sequential single requests
- `randomised_eval_20260310_174210.txt`: 91.9% +/- 1.2% overall accuracy across 3 sampled runs
- `attack_test_summary_20260302_132532.txt`: 95.0% overall accuracy (76/80) on a sampled attack-validation run

### Known limitations

The current saved evaluation artefacts show the weakest classes on rarer web and infiltration traffic:

| Class or signal | Current saved result |
|---|---|
| Web Attack - XSS | 53.3% mean accuracy in randomised evaluation; 60% in sampled attack test |
| Web Attack - Brute Force | 76.7% mean accuracy in randomised evaluation |
| Infiltration | 80.0% mean accuracy in randomised evaluation |
| Heartbleed | 90.0% mean accuracy in randomised evaluation; 80% in sampled attack test |
| Single-flow `/predict` latency | About 2262.9 ms average with SHAP enabled |

Batch inference is much faster than single-flow `/predict` calls because it skips per-request SHAP output and amortizes preprocessing/inference overhead.

## Testing and Evaluation

### API smoke tests

```bash
cd api
python test_client.py
```

### Benchmark suite

```bash
python benchmark_suite.py
```

Measures latency percentiles, batch throughput, and concurrent request handling. Results are written to `results/`.

### Randomised evaluation

```bash
python randomised_evaluation.py
```

Runs repeated sampled evaluations and writes aggregate accuracy summaries to `results/`.

### Attack detection validation

```bash
python test_attack_detection.py
```

Sends sampled CIC-IDS flows to the API and records per-class correctness plus latency.

### Demo traffic replay

```bash
python demo_traffic_simulator.py
```

Replays stored CIC-IDS flows to the API with keyboard controls for switching between attack mixes.

### Kali live testing

The `testing/` directory contains lab helpers for generating real attack traffic:

- `vulnerable_webapp.py` - intentionally insecure Flask app for SQLi, XSS, and brute-force testing
- `setup_services.py` - starts the vulnerable web app and optional FTP service
- `run_kali_tests.py` - semi-automated Kali attack runner
- `parse_kali_results.py` - correlates captured predictions with the planned attack windows
- `verify_interface.py` - helper for checking capture interfaces

## Alerting

The engine supports two layers of detection-related config inside `config/nids_config.yaml`:

- `alerts` controls engine-side behaviour such as confidence threshold and log-only mode
- `alerting` configures delivery channels such as email and Slack

### Email

```yaml
alerting:
  email:
    enabled: true
    smtp_server: smtp.gmail.com
    smtp_port: 587
    use_tls: true
    sender: nids-alerts@example.com
    password: ${NIDS_EMAIL_PASSWORD}
    recipients:
      - security-team@example.com
    min_confidence: 0.9
    cooldown_minutes: 5
    severity_threshold: medium
```

### Slack

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

Alerts include attack type, confidence, source/destination metadata, and flow context. Cooldowns help avoid flooding during sustained attacks.

## Configuration

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `NIDS_LOG_LEVEL` | `INFO` | API log level |
| `NIDS_REQUIRE_AUTH` | `false` | Enable API key auth |
| `NIDS_API_KEY` | none | Primary API key |
| `NIDS_API_KEYS` | none | Additional comma-separated API keys |
| `NIDS_RATE_LIMIT_ENABLED` | `true` | Enable or disable rate limiting |
| `NIDS_RATE_LIMIT` | `100/minute` | Default limit |
| `NIDS_PREDICT_RATE_LIMIT` | `60/minute` | `/predict` limit |
| `NIDS_BATCH_RATE_LIMIT` | `30/minute` | `/batch_predict` limit |
| `NIDS_DEBUG` | `false` | Enable uvicorn reload mode |
| `NIDS_CORS_ORIGINS` | `http://localhost:5001,http://localhost:3000` | Allowed CORS origins |
| `NIDS_API_URL` | `http://localhost:8000` | API URL for the dashboard and some helpers |
| `DASHBOARD_HOST` | `0.0.0.0` | Flask dashboard bind host |
| `DASHBOARD_PORT` | `5001` | Flask dashboard port |
| `REFRESH_INTERVAL` | `5` | Dashboard refresh interval in seconds |
| `FLASK_DEBUG` | `false` | Flask debug mode |
| `FLASK_SECRET_KEY` | auto-generated | Flask secret key |
| `NIDS_EMAIL_PASSWORD` | none | SMTP password |
| `NIDS_SLACK_WEBHOOK` | none | Slack webhook URL |

### Engine config

`config/nids_config.yaml` controls:

- capture interface, promiscuous mode, and BPF filter
- flow timeout and maximum flow limits
- API URL, timeout, and retry count
- engine-side alert threshold behaviour
- email and Slack channel settings
- log level and session log paths
- packet batch size, cleanup interval, and stats interval

## Notes on Repo State

- This checkout contains scripts and saved artefacts, not the older notebook set referenced in some historical docs.
- Models and processed arrays are already present in the repo.
- Prior analysis and explainability outputs are preserved under `figures/` and `results/`.

## Acknowledgements

- Dataset: [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)
- Citation: Sharafaldin, I., Lashkari, A.H., and Ghorbani, A.A. (2018). Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization. *ICISSP 2018*.

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE).
