# A machine learning based approach towards building a real-time Network Intrusion Detection System

## Problem Description
With the rising number of network-enabled devices connected to the internet such as mobile phones, IoT appliances, and vehicles, concerns about the security implications of using these devices are growing. The increase in the number and types of networked devices inevitably leads to a wider surface of attack, whereas the impact of successful attacks becomes increasingly severe as more critical responsibilities are assumed by these devices.

To identify and counter network attacks it is common to employ a combination of multiple systems in order to prevent attacks from happening or to detect and stop ongoing attacks if they cannot be prevented initially. These systems are usually comprised of an intrusion prevention system such as a firewall as the first layer of security, with intrusion detection systems representing the second layer. Should the intrusion prevention system be unable to prevent a network attack, it is the task of the detection system to identify malicious network traffic in order to stop the ongoing attack and keep the recorded network traffic data for later analysis. This data can subsequently be used to update the prevention system to allow for the detection of the specific network attack in the future. The need for intrusion detection systems is rising as absolute prevention against attacks is not possible due to the rapid emergence of new attack types.

Even though intrusion detection systems are an essential part of network security, many detection systems deployed today have a significant weakness as they rely on signature-based attack classification patterns. These patterns are able to detect the most common known attack patterns but have the drawback of being unable to detect novel attack types. To overcome this limitation, research in intrusion detection systems is focusing on more dynamic approaches based on machine learning and anomaly detection methods. In these systems, the normal network behaviour is learned by processing previously recorded benign data packets, which allows the system to identify new attack types by analysing network traffic for anomalous data flows.

This project aims to implement a classifier capable of identifying network traffic as either benign or malicious based on machine learning methodologies, and to deploy it as a complete real-time detection system with explainability, monitoring, and alerting capabilities.

## Data
The data used to train the classifier is taken from the [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) dataset provided by the Canadian Institute for Cybersecurity. It was created by capturing all network traffic during five days of operation inside a controlled network environment where realistic background traffic and different attack scenarios were conducted. As a result the dataset contains both benign network traffic as well as captures of the most common network attacks.

The dataset is comprised of raw network captures in pcap format as well as csv files created by using [CICFlowMeter](https://www.unb.ca/cic/research/applications.html#CICFlowMeter) containing 84 statistical features of the individual network flows combined with their corresponding labels. A network flow is defined as an aggregation of interrelated network packets identified by the following properties:
* Source IP
* Destination IP
* Source port
* Destination port
* Protocol

The dataset contains approximately 2.1 million individual network flows and covers the following attack scenarios:
* Brute Force (FTP-Patator, SSH-Patator)
* DoS (Slowloris, Slowhttptest, Hulk, GoldenEye)
* DDoS
* Heartbleed
* Web Attack (Brute Force, XSS, SQL Injection)
* Infiltration
* Botnet
* Portscan

## Approach
The goal of this project is to create a classifier capable of categorising network flows into 15 distinct attack types and benign traffic. The problem is understood as a supervised learning problem using the labels provided in the dataset. It is formulated as a multiclass classification problem differentiating between the individual classes of attacks provided in the dataset. A relevant subset of the features provided in the dataset will be used as predictors to classify individual network flows.

Traditional machine learning methods, specifically Random Forest and XGBoost, are applied to the problem in an ensemble configuration. Class imbalance is addressed using SMOTE, applied exclusively to the training set to prevent data leakage. Feature selection is performed using ANOVA F-test to reduce the feature space from 84 to the 50 most discriminative features. Model predictions are enhanced with SHAP-based explanations to provide feature-level attributions for every classification decision.

Beyond the classifier itself, the project implements a complete real-time detection pipeline: live packet capture via Scapy, 5-tuple bidirectional flow aggregation, in-line extraction of the 84 CIC-IDS2017 features, and asynchronous prediction submission to the classification service. Heuristic detectors (SYN flood and port scan) supplement the ML classifier with cross-flow pattern detection.

## Deliverables
The classifier is deployed and served via a REST API in conjunction with a real-time packet capture engine, a monitoring dashboard, and an alerting system.

The REST API provides the following functionality:
* an endpoint to submit individual network flows for classification with SHAP-based explainability
* an endpoint to submit batches of up to 1000 flows for efficient bulk prediction
* an endpoint exposing Prometheus metrics for operational monitoring
* health, statistics, and introspection endpoints for system observability
* optional API key authentication and per-endpoint rate limiting

The real-time engine provides:
* live packet capture on a configurable network interface with kernel-level BPF filtering
* 5-tuple bidirectional flow aggregation with configurable completion conditions
* feature extraction matching the 84 CIC-IDS2017 features used during training
* asynchronous prediction submission to the REST API via a thread pool
* session-based logging with separate engine, prediction, attack, and summary logs per run
* Email (SMTP) and Slack (webhook) alerting with severity-based thresholds and cooldown periods

To further showcase the project, a demonstration traffic simulator replays real CIC-IDS2017 flows against the running system to illustrate end-to-end detection across each attack category.

## Computational resources
The requirements regarding the computational resources to train the classifiers are given below:

| Category | Resource |
| -------- | -------- |
| CPU | Intel Core i7 processor |
| RAM | 16 GB |
| GPU | Not required |
| HDD | 20 GB (for dataset + model artifacts) |

## Classifier

The machine learning estimator created in this project follows a supervised approach and is trained using a soft-voting ensemble of two tree-based algorithms: a [Random Forest](https://en.wikipedia.org/wiki/Random_forest) classifier and an [XGBoost](https://xgboost.readthedocs.io/) gradient boosting classifier. The ensemble averages the probability distributions from both models to produce the final classification, combining Random Forest's variance reduction via bagging with XGBoost's bias reduction via boosting.

Random Forest is configured with 200 trees, a maximum depth of 20, and a minimum of 5 samples required to split a node. XGBoost is configured with up to 300 boosting rounds, a maximum tree depth of 10, a learning rate of 0.1, row and column sampling of 0.8, and early stopping after 10 rounds without validation loss improvement. The full set of chosen parameters and evaluation metrics can be examined in [`04_model_training_improved.py`](04_model_training_improved.py).

Predictions are enhanced with [SHAP](https://shap.readthedocs.io/) (SHapley Additive exPlanations) using the TreeExplainer implementation, which computes exact Shapley values for tree-based models. Each prediction returned by the API includes the top five contributing features along with their attribution values.

## Deployment Architecture

The deployment architecture of the complete NIDS-ML system is explained in detail in [`SYSTEM_EXPLANATION.md`](SYSTEM_EXPLANATION.md). A summary of the pipeline: trained models are served by a FastAPI prediction service on port 8000, which is consumed by both a real-time packet capture engine running locally and a Flask monitoring dashboard on port 5001. Prometheus scrapes the API's `/metrics` endpoint every 15 seconds, and Grafana provides visualisation over the collected time-series data via an 11-panel pre-built dashboard.

## Model Training and Deployment

The model can be trained and deployed locally using the provided Python scripts. Containerised deployment of the full stack (API + Prometheus + Grafana) is supported via Docker Compose.

### Installation

To install the necessary dependencies checkout the project and create a new Python virtual environment.

```
python -m venv .venv
```

Afterwards activate the environment and install the project dependencies.

```
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

pip install -r api/requirements.txt
pip install -r src/requirements.txt
pip install -r dashboard/requirements.txt
```

Live packet capture requires [Npcap](https://npcap.com/) on Windows or libpcap on Linux/Mac.

### Dataset Creation

To create the dataset for training, download the raw [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) dataset and run the preprocessing script:

```
python 02_data_preprocessing.py
```

This reads the source CSVs, consolidates attack labels, cleans problematic values (infinity, NaN), and saves the cleaned dataset to `data/processed/cicids2017_cleaned.csv`.

Then train the models:

```
python 04_model_training_improved.py
```

The training script applies StandardScaler normalisation, performs ANOVA-based feature selection (84 features reduced to 50), applies SMOTE to the training set to address class imbalance, and trains the Random Forest + XGBoost ensemble. Model artifacts are saved to `models/`.

### Local Mode

To train the model using the default parameters and dataset locations, use the following command:

```
python 04_model_training_improved.py
```

Upon completion of the training process the model artifacts can be found in the `models/` directory:

| File | Purpose |
| ---- | ------- |
| `random_forest_model.pkl` | Random Forest classifier (~140 MB) |
| `xgboost_model.pkl` | XGBoost classifier (~7 MB) |
| `scaler.pkl` | StandardScaler fitted on training data |
| `feature_selector.pkl` | SelectKBest (84 to 50 features) |
| `label_encoder.pkl` | Maps 16 classes to numeric labels |
| `training_config.pkl` | Training metadata |

To deploy the model locally, start the FastAPI prediction service:

```
python run_nids.py --api-only
```

The API will be accessible at `http://localhost:8000` with interactive Swagger documentation at `http://localhost:8000/docs`. To launch both the API and the real-time packet capture engine together (requires administrator privileges):

```
python run_nids.py
```

To specify a network interface explicitly:

```
python run_nids.py -i Ethernet
```

Session logs for each run are saved to `logs/sessions/YYYYMMDD_HHMMSS/` with separate engine, prediction, attack, and summary files.

### Docker

The complete stack (API, Prometheus, and Grafana) can also be deployed as Docker containers using the provided Docker Compose configuration:

```
docker-compose up -d
```

After startup the following services become available:

| Service | URL | Credentials |
| ------- | --- | ----------- |
| NIDS API | http://localhost:8000 | — |
| Swagger Docs | http://localhost:8000/docs | — |
| Flask Dashboard | http://localhost:5001 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / nids-admin |

To stop the stack:

```
docker-compose down
```

## Using the Classifier

The classifier can be accessed directly through the REST API exposed by the prediction service. Once the API is running, predictions can be submitted either as individual flows or as batches of up to 1000 flows per request.

### REST API

To invoke the REST API, the following command can be used to submit a prediction request for a given network flow:

```
curl -X POST \
  http://localhost:8000/predict \
  -H 'Accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"features": {"Src Port": 443, "Dst Port": 80, "Protocol": 6, "Flow Duration": 119759145, "Total Fwd Packet": 75837, "Total Bwd packets": 0, ...}}'
```

A successful prediction response includes the predicted class, confidence score, full probability distribution across all 16 classes, and the top five contributing features identified via SHAP:

```json
{
  "prediction": "DDoS",
  "confidence": 0.9847,
  "is_attack": true,
  "probabilities": {"BENIGN": 0.012, "DDoS": 0.985, "DoS Hulk": 0.002, ...},
  "explanation": [
    {"feature": "Flow Packets/s", "value": 15234.5, "contribution": 0.42},
    {"feature": "Dst Port", "value": 80.0, "contribution": 0.28}
  ]
}
```

The full API reference, including the batch prediction endpoint, health checks, statistics, and Prometheus metrics, is available through the interactive Swagger documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

### Demonstration Traffic Simulator

For convenience, an interactive demonstration simulator is provided that replays real CIC-IDS2017 flows against the running API to showcase end-to-end detection across each attack category:

```
python demo_traffic_simulator.py
```

The simulator provides keyboard controls to switch between attack scenarios (DDoS, Port Scan, DoS Hulk, Brute Force, Web Attacks, Botnet, Infiltration), adjust replay speed, and pause or resume traffic. A live terminal display tracks overall accuracy, attack detection rate, and recent predictions in real time.

## Acknowledgements

- **Dataset**: [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) — Canadian Institute for Cybersecurity, University of New Brunswick
- **Citation**: Sharafaldin, I., Lashkari, A.H., & Ghorbani, A.A. (2018). Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization. *ICISSP 2018*.

## License

This project is licensed under the GNU General Public License v3.0 — see the [LICENSE](LICENSE) file for details.
