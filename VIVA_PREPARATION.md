# NIDS-ML: Complete Project Guide & Viva Preparation

This document covers every component of the system, the reasoning behind every decision, and answers to questions an examiner is likely to ask.

---

## TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [Dataset: CIC-IDS2017](#2-dataset-cic-ids2017)
3. [Data Preprocessing Pipeline](#3-data-preprocessing-pipeline)
4. [Feature Engineering](#4-feature-engineering)
5. [Model Training](#5-model-training)
6. [The Ensemble Approach](#6-the-ensemble-approach)
7. [Explainability (SHAP & LIME)](#7-explainability-shap--lime)
8. [FastAPI Prediction Service](#8-fastapi-prediction-service)
9. [Real-Time NIDS Engine](#9-real-time-nids-engine)
10. [Packet Capture](#10-packet-capture)
11. [Flow Aggregation](#11-flow-aggregation)
12. [Feature Extraction (84 Features)](#12-feature-extraction-84-features)
13. [Heuristic Detectors](#13-heuristic-detectors)
14. [Alerting System](#14-alerting-system)
15. [Dashboard & Monitoring](#15-dashboard--monitoring)
16. [Docker & Deployment](#16-docker--deployment)
17. [Testing & Evaluation](#17-testing--evaluation)
18. [Security Features](#18-security-features)
19. [Known Limitations & Critical Analysis](#19-known-limitations--critical-analysis)
20. [Future Work](#20-future-work)
21. [Viva Questions & Answers](#21-viva-questions--answers)

---

## 1. PROJECT OVERVIEW

### What it is
A Machine Learning-based Network Intrusion Detection System (NIDS) that classifies network traffic flows into 15 attack types + benign using a Random Forest + XGBoost ensemble. It includes a complete pipeline from raw data to real-time detection with monitoring and alerting.

### Architecture (data flow)
```
CIC-IDS2017 Raw CSVs (5 days)
    |
    v
[02_data_preprocessing.py] -- Clean, consolidate labels, remove NaN/Inf
    |
    v
cicids2017_cleaned.csv (2.1M flows, 84 features)
    |
    v
[03_feature_engineering.py] -- Scale, select top 50, SMOTE balance
    |
    v
X_train/val/test .npy + scaler.pkl + feature_selector.pkl + label_encoder.pkl
    |
    v
[04_model_training_improved.py] -- Train RF + XGBoost, evaluate, save
    |
    v
random_forest_model.pkl + xgboost_model.pkl
    |
    +---> [api/app.py: FastAPI] ---- /predict, /batch_predict, /stats, /metrics
    |          |
    |          +---> [dashboard/app.py: Flask] ---- Real-time status page
    |          |
    |          +---> [Prometheus] --> [Grafana] ---- 11-panel monitoring dashboard
    |
    +---> [run_nids.py] -- Launches API + Engine together
              |
              v
         [src/nids_engine.py] -- Orchestrator
              |
              +---> [packet_capture.py] -- Scapy, BPF filter, thread-safe queue
              |
              +---> [flow_aggregator.py] -- 5-tuple grouping, bidirectional
              |
              +---> [feature_extractor.py] -- Computes 84 CIC-IDS2017 features
              |
              +---> HTTP POST to /predict --> Gets classification
              |
              +---> [alerting.py] -- Email (SMTP) + Slack webhooks
```

### Tech stack
| Layer | Technology |
|-------|-----------|
| ML Models | scikit-learn (Random Forest), XGBoost |
| Data Processing | Pandas, NumPy |
| Oversampling | SMOTE (imbalanced-learn) |
| Feature Selection | SelectKBest with ANOVA F-test |
| Explainability | SHAP (TreeExplainer), LIME |
| API | FastAPI + Uvicorn |
| Packet Capture | Scapy |
| Dashboard | Flask |
| Monitoring | Prometheus + Grafana |
| Containerisation | Docker + Docker Compose |
| Authentication | API Key (X-API-Key header) |
| Rate Limiting | slowapi (fixed-window, IP-based) |

---

## 2. DATASET: CIC-IDS2017

### What it is
The Canadian Institute for Cybersecurity Intrusion Detection Dataset 2017. Created at the University of New Brunswick by capturing real network traffic over 5 days (Monday-Friday), with benign background traffic and specific attacks executed on each day.

### Why this dataset
- **Labelled**: Every flow has a ground-truth label (attack type or benign)
- **Realistic**: Uses real network infrastructure, not synthetic
- **Comprehensive**: 15 distinct attack types covering major NIDS categories
- **Widely cited**: Standard benchmark in IDS research (600+ citations)
- **Feature-rich**: Pre-computed flow-level features using CICFlowMeter

### Dataset statistics
- **Total flows**: ~2.1 million after cleaning
- **Features**: 84 network flow features
- **Classes**: 16 (BENIGN + 15 attack types)

### Attack types by day
| Day | Attacks |
|-----|---------|
| Monday | Benign only (baseline) |
| Tuesday | FTP-Patator, SSH-Patator |
| Wednesday | DoS Slowloris, DoS Slowhttptest, DoS Hulk, DoS GoldenEye, Heartbleed |
| Thursday | Web Attack (Brute Force, XSS, SQL Injection), Infiltration |
| Friday | Botnet, Portscan, DDoS |

### Class imbalance (the problem)
The dataset is heavily imbalanced:
- BENIGN: ~1.9M flows (dominant)
- DDoS: ~128K
- Portscan: ~158K
- DoS Hulk: ~231K
- Web Attack - XSS: 652 flows (tiny)
- Web Attack - Brute Force: 1507 flows (tiny)
- Heartbleed: 11 flows (extremely rare)
- Infiltration: 36 flows (extremely rare)

This imbalance is why SMOTE is needed and why some classes underperform.

---

## 3. DATA PREPROCESSING PIPELINE

**File**: `02_data_preprocessing.py` (238 lines)

### Step-by-step
1. **Load 5 CSV files** (one per day) from raw CIC-IDS2017 dataset
2. **Combine** into single DataFrame
3. **Consolidate labels**: Remove " - Attempted" suffix (e.g., "SSH-Patator - Attempted" becomes "SSH-Patator") -- reduces unnecessary class fragmentation
4. **Clean problematic values**:
   - Replace infinity values with column median
   - Fix negative header lengths by taking absolute value
   - Fill NaN values with column median
5. **Remove non-feature columns**: `id`, `Flow ID`, `Src IP`, `Dst IP`, `Timestamp`, `Attempted Category` -- these are identifiers, not features
6. **Visualise** class distribution bar chart
7. **Save** to `data/processed/cicids2017_cleaned.csv`

### Why median (not mean) for imputation
Median is robust to outliers. Network flow features often have extreme outliers (e.g., a single massive packet), so median preserves the typical value better than mean.

### Why remove IP addresses
IP addresses are identifiers, not behavioural features. A model trained on specific IPs would memorise the lab setup, not learn attack patterns. This prevents overfitting to the CIC-IDS2017 network topology.

---

## 4. FEATURE ENGINEERING

**File**: `03_feature_engineering.py` (350 lines)

### Scaling: StandardScaler
- Transforms each feature to zero mean and unit variance
- Formula: `z = (x - mean) / std`
- Fitted on training data only (prevents data leakage)
- Why: Many ML algorithms (including tree ensembles) benefit from normalised features. XGBoost specifically handles unscaled data, but scaling ensures consistent behaviour across the ensemble.

### Feature selection: SelectKBest with ANOVA F-test
- Scores each feature using the F-statistic from a one-way ANOVA
- Selects top K=50 features from 84
- Why 50: Balances dimensionality reduction (removes noise) with information retention. 50 keeps 60% of features while removing low-signal ones.
- Why ANOVA: It measures how much each feature's distribution differs across classes. Features with high F-scores discriminate well between attack types.

### The 50 selected features (in order)
Dst Port, Protocol, Flow Duration, Fwd Packet Length Min, Fwd Packet Length Mean, Bwd Packet Length Max, Bwd Packet Length Min, Bwd Packet Length Mean, Bwd Packet Length Std, Flow Packets/s, Flow IAT Std, Flow IAT Max, Fwd IAT Total, Fwd IAT Std, Fwd IAT Max, Bwd IAT Total, Bwd IAT Mean, Bwd IAT Std, Bwd IAT Max, Fwd PSH Flags, Fwd RST Flags, Bwd RST Flags, Fwd Packets/s, Bwd Packets/s, Packet Length Min, Packet Length Max, Packet Length Mean, Packet Length Std, Packet Length Variance, FIN Flag Count, SYN Flag Count, RST Flag Count, Down/Up Ratio, Average Packet Size, Fwd Segment Size Avg, Bwd Segment Size Avg, Bwd Bulk Rate Avg, Subflow Fwd Packets, Subflow Fwd Bytes, Subflow Bwd Bytes, FWD Init Win Bytes, Fwd Act Data Pkts, Fwd Seg Size Min, Active Mean, Active Std, Active Max, Active Min, Idle Mean, Idle Std, Idle Max.

### SMOTE (Synthetic Minority Over-sampling Technique)
- **What**: Generates synthetic samples for minority classes by interpolating between existing samples and their k-nearest neighbours
- **Parameters**: `sampling_strategy='not majority'`, `k_neighbors=5`, `random_state=42`
- **Applied to training set only** -- this is critical. If SMOTE were applied before the train/test split, synthetic samples derived from test data would leak into training, inflating accuracy.
- **Why not random oversampling**: Random oversampling duplicates existing samples, causing overfitting. SMOTE creates novel samples in feature space, improving generalisation.

### Data split
- Training: 70%
- Validation: 15%
- Test: 15%
- Method: Stratified split (preserves class proportions in each set)
- Random state: 42

### Label encoding
- LabelEncoder maps class names to integers 0-15
- Saved as `models/label_encoder.pkl` for consistent decoding at inference time

---

## 5. MODEL TRAINING

**File**: `04_model_training_improved.py` (915 lines)

### Why Random Forest + XGBoost

**Random Forest**:
- Bagging ensemble of decision trees
- Trains multiple independent trees on random subsets of data and features
- Robust to overfitting, handles high-dimensional data well
- Naturally handles multi-class problems
- Gives feature importance for free

**XGBoost**:
- Gradient boosting ensemble
- Trains trees sequentially, each correcting the previous tree's errors
- Generally higher accuracy than RF on tabular data
- Built-in regularisation (L1/L2)
- Early stopping prevents overfitting

**Why both**: They make different types of errors. RF is more robust (lower variance), XGBoost is more accurate (lower bias). Ensembling them improves overall performance.

### Exact hyperparameters

**Random Forest:**
```
n_estimators = 200       # Number of trees
max_depth = 20           # Maximum tree depth
min_samples_split = 5    # Minimum samples to split a node
min_samples_leaf = 2     # Minimum samples in a leaf
max_features = 'sqrt'    # Features considered per split = sqrt(50) ~ 7
random_state = 42
n_jobs = 2               # Parallel threads
```

**XGBoost:**
```
n_estimators = 300       # Maximum boosting rounds
max_depth = 10           # Maximum tree depth
learning_rate = 0.1      # Step size shrinkage
subsample = 0.8          # Row sampling per tree (80%)
colsample_bytree = 0.8   # Feature sampling per tree (80%)
objective = 'multi:softprob'  # Multi-class with probabilities
eval_metric = 'mlogloss'      # Log loss for evaluation
early_stopping_rounds = 10    # Stop if no improvement for 10 rounds
random_state = 42
n_jobs = 2
```

### Why these specific values
- **max_depth (RF=20, XGB=10)**: RF uses deeper trees because bagging controls variance. XGBoost uses shallower trees because boosting accumulates depth across iterations.
- **n_estimators (RF=200, XGB=300)**: More trees generally help until diminishing returns. XGBoost has early stopping so 300 is a maximum, not a guarantee.
- **learning_rate=0.1**: Standard default. Lower values (0.01) give better results but need more trees and training time.
- **subsample/colsample_bytree=0.8**: Introduces randomness to prevent overfitting, similar to RF's random subsets.

### Cross-validation
- Method: StratifiedKFold (preserves class proportions)
- Folds: 3 (reduced from 5 for memory efficiency)
- Primary metric: f1_macro

### Why f1_macro (not accuracy)
With class imbalance, accuracy is misleading. A model that predicts "BENIGN" for everything gets ~90% accuracy. F1_macro averages F1 scores across all classes equally, so poor performance on rare classes (XSS, Infiltration) properly lowers the score.

### Evaluation results

**On validation set:**
| Model | Accuracy | F1 Macro | F1 Weighted |
|-------|----------|----------|-------------|
| Random Forest | 99.40% | 0.8732 | 99.40% |
| XGBoost | 99.45% | 0.9034 | 99.45% |

**On test set (XGBoost):**
| Metric | Value |
|--------|-------|
| Accuracy | 99.44% |
| F1 Macro | 0.8337 |
| Precision Macro | 0.8780 |
| Recall Macro | 0.8123 |

**Key observation**: Accuracy (99.44%) vs F1 Macro (0.8337) -- the gap confirms class imbalance effects. The model is excellent on majority classes but struggles on rare ones.

---

## 6. THE ENSEMBLE APPROACH

### How it works
Soft voting: average the probability distributions from both models.

```python
rf_probs = random_forest.predict_proba(features)    # shape: (1, 16)
xgb_probs = xgboost.predict_proba(features)          # shape: (1, 16)
ensemble_probs = (rf_probs + xgb_probs) / 2.0        # average
prediction = argmax(ensemble_probs)                   # highest probability class
confidence = max(ensemble_probs)                      # confidence score
```

### Why soft voting (not hard voting)
Hard voting takes the majority class from each model's prediction. Soft voting averages probabilities, which captures uncertainty. For example:
- RF says: DDoS (0.6), Benign (0.4)
- XGBoost says: DDoS (0.55), DoS Hulk (0.45)
- Soft vote: DDoS (0.575) -- correct, with nuance
- Hard vote: DDoS (2 votes) -- same result but loses the probability information

Soft voting is particularly useful when models disagree on the second-most-likely class.

### Model agreement tracking
The API tracks when RF and XGBoost agree vs disagree. Disagreement indicates uncertain predictions, which is logged for monitoring.

---

## 7. EXPLAINABILITY (SHAP & LIME)

### SHAP (SHapley Additive exPlanations)
- **Method**: TreeExplainer (exact Shapley values for tree models)
- **What it does**: For each prediction, shows how much each feature pushed the prediction towards or away from a given class
- **Implementation**: Computed per prediction in the API, returns top 5 contributing features
- **Example output**: "Dst Port contributed +0.23 towards DDoS, Flow Duration contributed -0.15 away from DDoS"
- **Why SHAP**: Based on game theory (Shapley values), provides consistent and locally accurate feature attributions. Unlike feature importance, SHAP explains individual predictions, not just global patterns.

### LIME (Local Interpretable Model-agnostic Explanations)
- **Method**: Perturbs input features, observes prediction changes, fits a local linear model
- **Used for**: Generating explanation visualisations in `figures/lime/`
- **Difference from SHAP**: SHAP is exact for tree models; LIME is approximate but model-agnostic

### Why both
SHAP and LIME provide different perspectives. SHAP gives mathematically rigorous attributions; LIME gives intuitive "what if" explanations. Including both demonstrates understanding of the explainability landscape. The XAI comparison results are in `results/xai_comparison.csv`.

---

## 8. FASTAPI PREDICTION SERVICE

**File**: `api/app.py` (~950 lines)

### Endpoints

| Endpoint | Method | Purpose | Rate Limit |
|----------|--------|---------|------------|
| `/` | GET | Welcome + API info | 100/min |
| `/health` | GET | Model status, uptime, memory | 100/min |
| `/stats` | GET | Prediction counts, attack rate, avg latency | 100/min |
| `/predict` | POST | Single flow prediction + SHAP explanation | 60/min |
| `/batch_predict` | POST | Up to 1000 flows at once | 30/min |
| `/features` | GET | List 84 feature names | 100/min |
| `/classes` | GET | List 16 class labels | 100/min |
| `/metrics` | GET | Prometheus metrics (counters, histograms) | None |
| `/security` | GET | Auth + rate limit status | 100/min |
| `/docs` | GET | Swagger UI (auto-generated) | None |

### Prediction pipeline (what happens on /predict)
1. Receive JSON with 84 feature values
2. Validate all 84 feature names are present
3. Build feature vector in correct order (matching training)
4. Apply StandardScaler (loaded from `models/scaler.pkl`)
5. Apply SelectKBest to reduce 84 -> 50 features (loaded from `models/feature_selector.pkl`)
6. Get probability predictions from both RF and XGBoost
7. Average probabilities (soft voting ensemble)
8. Decode class label using LabelEncoder
9. Compute SHAP explanation (top 5 features)
10. Return prediction, confidence, model agreement, SHAP explanations

### Why FastAPI
- Async support (important for non-blocking predictions)
- Auto-generated Swagger documentation
- Pydantic validation (catches malformed requests before hitting the model)
- High performance (Uvicorn ASGI server)
- Native support for dependency injection (used for auth)

---

## 9. REAL-TIME NIDS ENGINE

**File**: `src/nids_engine.py` (~870 lines)

### Architecture
The engine runs as a continuous loop with a 100ms tick rate:

```
Every 100ms:
  1. Drain packets from capture queue
  2. Feed packets into flow aggregator
  3. Check for completed flows (TCP FIN/RST or timeout)
  4. Extract 84 features from each completed flow
  5. Submit prediction request to API (non-blocking, via thread pool)
  6. Run heuristic detectors (SYN flood, port scan)
  7. Handle completed predictions (log, alert)
  8. Periodic cleanup of expired flows
```

### Threading model
- **Main thread**: Packet processing loop (capture -> aggregate -> extract -> submit)
- **Capture thread**: Scapy runs in a separate thread to avoid blocking the main loop
- **Prediction thread pool**: `ThreadPoolExecutor(max_workers=4)` -- predictions are HTTP calls to the API, so they're I/O bound and benefit from threading
- **Why 4 workers**: Matches typical CPU core count, prevents overwhelming the API

### Session logging
Each run creates a timestamped directory (`logs/sessions/YYYYMMDD_HHMMSS/`) with:
- `engine.log` -- operational events
- `predictions.log` -- every classification result
- `attacks.log` -- detected attacks only
- `summary.log` -- session statistics on shutdown

### Startup sequence (run_nids.py)
1. Create session directory
2. Start API as subprocess (port 8000)
3. Poll `/health` every second until ready (up to 30 seconds)
4. Start NIDS engine as subprocess
5. Pipe output from both to log files + console
6. On Ctrl+C: gracefully terminate both processes

---

## 10. PACKET CAPTURE

**File**: `src/packet_capture.py`

### How it works
- Uses Scapy's `sniff()` function in a background thread
- BPF (Berkeley Packet Filter) at kernel level: `"ip"` -- captures only IP packets, drops ARP/broadcast at the kernel, not in Python
- Promiscuous mode: enabled by default (captures all traffic on the network segment, not just traffic addressed to this machine)
- Packets are parsed and placed in a thread-safe queue (max 50,000 packets)
- When queue is full, oldest packets are dropped (prevents memory exhaustion)

### What gets extracted per packet
| Field | Source |
|-------|--------|
| timestamp | `time.time()` |
| src_ip, dst_ip | IP layer |
| src_port, dst_port | TCP/UDP layer |
| protocol | IP protocol number (6=TCP, 17=UDP, 1=ICMP) |
| length | Total IP packet length |
| payload_length | Transport layer payload size |
| tcp_flags | TCP flag string (F, S, R, P, A, U, E, C) |
| header_length | IP header length |
| window_size | TCP window size |
| icmp_type, icmp_code | ICMP fields (if applicable) |

### Why Scapy (not tcpdump/tshark)
- Pure Python: no external binary dependencies
- Programmatic access: parsed packet fields, not raw text
- Cross-platform: works on Windows (with Npcap) and Linux
- Fine-grained control: custom callbacks, filters, timeouts

### Requires admin privileges
Raw packet capture requires elevated permissions on all operating systems. On Windows, Npcap must be installed.

---

## 11. FLOW AGGREGATION

**File**: `src/flow_aggregator.py` (512 lines)

### What is a flow
A flow is a sequence of packets sharing the same 5-tuple: (source IP, destination IP, source port, destination port, protocol). It represents a single network conversation.

### Bidirectional tracking
Flows are bidirectional. A packet from A->B and a response from B->A belong to the same flow. The aggregator uses a canonical key:

```
Forward:  (src_ip, dst_ip, src_port, dst_port, protocol)
Backward: (dst_ip, src_ip, dst_port, src_port, protocol) -> maps to same flow
```

Packets are tagged as "forward" or "backward" based on which direction matches the flow's initiator.

### Flow completion conditions
A flow is considered complete when any of these occur:
1. **TCP FIN exchange**: Both forward and backward FIN flags seen (graceful close)
2. **TCP RST**: Any RST flag (connection reset)
3. **Max packets**: 1000 packets reached (safety cap)
4. **Timeout**: No new packet for 10 seconds (configurable)

### What's tracked per flow
- Forward/backward packet lists (timestamps, lengths, payload lengths)
- Per-direction flag counts: FIN, SYN, RST, PSH, ACK, URG, ECE, CWR
- Initial TCP window sizes (forward and backward)
- ICMP type/code
- Active/idle periods (idle threshold: 1.0 second)
- Total duration in microseconds

### Max simultaneous flows: 100,000
Safety limit to prevent memory exhaustion. If reached, oldest flows are force-completed.

---

## 12. FEATURE EXTRACTION (84 FEATURES)

**File**: `src/feature_extractor.py` (~500 lines)

### Complete feature list with computation method

**Basic (4 features)**:
1. `Src Port` -- Source port number
2. `Dst Port` -- Destination port number
3. `Protocol` -- IP protocol (6=TCP, 17=UDP, 1=ICMP)
4. `Flow Duration` -- Last packet time - first packet time (microseconds)

**Packet counts (2)**:
5. `Total Fwd Packet` -- Count of forward packets
6. `Total Bwd packets` -- Count of backward packets

**Packet lengths (10)**:
7-8. Total length of forward/backward packets (sum of lengths)
9-12. Forward packet length: max, min, mean, std
13-16. Backward packet length: max, min, mean, std

**Flow rates (2)**:
17. `Flow Bytes/s` -- Total bytes / duration in seconds
18. `Flow Packets/s` -- Total packets / duration in seconds

**Inter-Arrival Time - Flow (4)**:
19-22. Flow IAT: mean, std, max, min (time between consecutive packets, all directions)

**Inter-Arrival Time - Forward (5)**:
23-27. Fwd IAT: total (sum), mean, std, max, min

**Inter-Arrival Time - Backward (5)**:
28-32. Bwd IAT: total, mean, std, max, min

**TCP Flags per direction (6)**:
33-34. PSH flags: forward, backward
35-36. URG flags: forward, backward
37-38. RST flags: forward, backward

**Header lengths (2)**:
39-40. Header length: forward (sum), backward (sum)

**Packets per second (2)**:
41. Fwd Packets/s -- Forward packets / duration
42. Bwd Packets/s -- Backward packets / duration

**Overall packet length stats (5)**:
43-47. All packets combined: min, max, mean, std, variance

**Total TCP flag counts (8)**:
48-55. FIN, SYN, RST, PSH, ACK, URG, CWR, ECE (total across both directions)

**Ratios and averages (4)**:
56. Down/Up Ratio -- backward packets / forward packets
57. Average Packet Size -- mean of all packet lengths
58. Fwd Segment Size Avg -- mean of forward lengths
59. Bwd Segment Size Avg -- mean of backward lengths

**Bulk features (6)**: All set to 0.0
60-65. Fwd/Bwd bytes/bulk avg, packet/bulk avg, bulk rate avg
(These require session-level tracking not available in real-time capture)

**Subflow features (4)**:
66-69. Subflow fwd/bwd packets and bytes (equal to total in single-flow analysis)

**TCP window sizes (2)**:
70. FWD Init Win Bytes -- TCP window size of first forward packet
71. Bwd Init Win Bytes -- TCP window size of first backward packet

**Active data packets (2)**:
72. Fwd Act Data Pkts -- Forward packets with payload > 0
73. Fwd Seg Size Min -- Minimum non-zero payload length

**Active/Idle times (8)**: All in microseconds
74-77. Active: mean, std, max, min
78-81. Idle: mean, std, max, min
(Active = period with packets arriving; Idle = gap > 1 second)

**ICMP (2)**:
82. ICMP Code
83. ICMP Type

**TCP time (1)**:
84. Total TCP Flow Time -- Duration if TCP, else 0

### Safety wrappers
All statistical functions use safe_ wrappers:
- `safe_divide(a, b)` -- returns 0.0 if b is 0
- `safe_mean(lst)` -- returns 0.0 if list is empty
- `safe_stdev(lst)` -- returns 0.0 if < 2 values
- `safe_max/min(lst)` -- returns 0.0 if empty

This prevents crashes from edge cases (single-packet flows, empty backward direction, etc.).

---

## 13. HEURISTIC DETECTORS

**File**: `src/nids_engine.py`

These supplement the ML model with rule-based detection for specific attack patterns:

### SYN Flood Detector
- **Logic**: Tracks flows where `RST flags > 0 AND SYN flags > 0 AND total packets <= 5` (incomplete TCP handshakes)
- **Window**: 10 seconds (sliding)
- **Threshold**: 20 such flows within the window
- **Why**: SYN floods create many half-open connections that are RST-terminated. The ML model may not catch these if they're too short to form meaningful flow features.

### Port Scan Detector
- **Logic**: Tracks distinct (dst_ip, dst_port) pairs per source IP for flows with `total packets <= 3`
- **Window**: 30 seconds (sliding)
- **Threshold**: 15 distinct destination ports
- **Why**: Port scans probe many ports with minimal packets. Individual probe flows may look benign; the pattern across flows reveals the scan.

### Why heuristics alongside ML
The ML model classifies individual flows. Some attacks are only detectable by looking at patterns *across* flows (many short connections from same source). Heuristics complement the ML model by detecting these cross-flow patterns.

---

## 14. ALERTING SYSTEM

**File**: `src/alerting.py`

### Channels
1. **Email**: SMTP with TLS (Gmail, Outlook, or custom server)
2. **Slack**: Webhook API (posts to specified channel)

Both disabled by default (`log_only: true` in config).

### Severity mapping
| Attack | Severity |
|--------|----------|
| DDoS, Heartbleed, SQL Injection, Infiltration | Critical |
| DoS Hulk, DoS GoldenEye, Web Attack - XSS, Botnet | High |
| DoS Slowloris, DoS Slowhttptest, FTP/SSH-Patator, Brute Force | Medium |
| Portscan, BENIGN | Low |

### Cooldown logic
Prevents alert flooding. After sending an alert for a specific attack type, no more alerts for that type until the cooldown expires.
- Email cooldown: 5 minutes
- Slack cooldown: 2 minutes

### Confidence thresholds
- Email: Only alert if confidence >= 0.9 (high bar, fewer false alarms)
- Slack: Only alert if confidence >= 0.8 (lower bar, more responsive)

### Environment variable substitution
Passwords and webhook URLs use `${VAR_NAME}` syntax in config, resolved at runtime from environment variables. This keeps secrets out of config files.

---

## 15. DASHBOARD & MONITORING

### Flask Dashboard (`dashboard/app.py`, port 5001)
- Lightweight real-time status page
- Polls the API `/stats` endpoint every 5 seconds
- Shows: total predictions, attack count, attack rate, recent predictions
- Dark theme (Catppuccin Mocha colour scheme)
- Single-page design (no navigation needed)

### Prometheus (port 9090)
- Scrapes `/metrics` endpoint every 15 seconds
- Custom metrics exposed by the API:
  - `nids_predictions_total` (counter)
  - `nids_attacks_detected_total` (counter)
  - `nids_prediction_latency_seconds` (histogram)
  - `nids_models_loaded` (gauge)
- Retention: 30 days

### Grafana (port 3000)
- Auto-provisioned with Prometheus datasource
- Pre-built dashboard with 11 panels:
  - Prediction rate over time
  - Attack detection rate
  - Prediction latency histogram
  - Attack type distribution
  - Model agreement rate
  - Active flows count
  - Error rate
  - And more

### Why three monitoring systems
Each serves a different purpose:
- **Flask dashboard**: Quick operational overview, no setup needed
- **Prometheus**: Metric collection and storage, time-series queries
- **Grafana**: Rich visualisation, alerting rules, historical analysis

---

## 16. DOCKER & DEPLOYMENT

### Dockerfile (multi-stage build)
**Stage 1 (builder)**: Installs all Python dependencies into a virtual environment
**Stage 2 (production)**: Copies only the venv and application code

This reduces the final image size by excluding build tools (gcc, pip cache, etc.).

- Base image: `python:3.11-slim`
- Non-root user: `nids` (UID 1000) -- security best practice
- Health check: `curl http://localhost:8000/health`
- Exposed port: 8000

### Docker Compose (3 services)
```yaml
nids-api:      port 8000, CPU: 2 cores, RAM: 4GB
prometheus:    port 9090, CPU: 0.5 cores, RAM: 512MB
grafana:       port 3000, CPU: 0.5 cores, RAM: 256MB
```

- Health check ordering: Prometheus waits for API to be healthy before starting
- Models mounted read-only (`./models:/app/models:ro`)
- Logs persisted via volume mount

### Why Docker
- Reproducible environment (no "works on my machine" problems)
- Easy deployment to any server
- Resource limits prevent the API from consuming all system resources
- Service orchestration (health checks, restart policies, dependency ordering)

---

## 17. TESTING & EVALUATION

### test_attack_detection.py
- Loads real CIC-IDS2017 samples for each attack type
- Sends them to the API via /predict
- Measures per-class accuracy, precision, recall
- Configurable: `--attack DDoS --samples 5 --batch`

### benchmark_suite.py
4 phases:
1. **Single prediction latency**: 30 requests, measures p50/p95/p99
2. **Batch throughput**: Tests batch sizes 10, 50, 100, 500
3. **Concurrent throughput**: Tests 4, 8, 16 threads
4. **Single vs batch comparison**: Quantifies the speedup

**Results (March 2026):**
- Single prediction: p50 = 2257ms, p99 = 2427ms
- Batch of 500: 219 flows/sec (4.57ms per flow)
- Batch speedup: 98.3x over sequential

### randomised_evaluation.py
- Runs N evaluation rounds with different random seeds
- Reports mean accuracy +/- standard deviation with 95% confidence interval
- Addresses the concern that a single test split may not be representative

**Results (3 runs, 10 samples/class):**
- Overall: **91.9% +/- 1.2%**
- 95% CI: [90.5%, 93.3%]
- Per-class: BENIGN, DDoS, DoS variants, Botnet, Patators = 100%
- Weak: Web Attack - XSS = 53.3%, Brute Force = 76.7%

### demo_traffic_simulator.py
- Interactive replay of CIC-IDS2017 flows through the API
- Keyboard controls to switch between attack types
- Real-time accuracy tracking and terminal UI
- Used for viva demonstration

### Kali Linux testing (testing/ directory)
- `vulnerable_webapp.py`: Flask app with intentional XSS, SQL injection, brute-force vulnerabilities
- `setup_services.py`: Starts FTP/SSH services for brute-force targets
- `run_kali_tests.py`: Guides attack execution from Kali VM
- `parse_kali_results.py`: Correlates NIDS predictions with attack windows

---

## 18. SECURITY FEATURES

### API Authentication
- Optional X-API-Key header authentication
- Supports multiple keys via `NIDS_API_KEYS` environment variable
- Constant-time comparison (`hmac.compare_digest`) prevents timing attacks
- Disabled by default for development convenience

### Rate Limiting
- slowapi library (built on limits)
- IP-based with proxy header support (X-Forwarded-For, X-Real-IP)
- Per-endpoint limits: 60/min for predict, 30/min for batch, 100/min general
- Fixed-window strategy
- In-memory storage (would use Redis for multi-instance deployment)

### Input Validation
- Pydantic models validate all request payloads
- Exact 84 features required (no more, no less)
- Feature names validated against expected list
- Batch size capped at 1000 flows

### Docker Security
- Non-root container user
- Read-only model volume mount
- Resource limits (CPU, memory)

---

## 19. KNOWN LIMITATIONS & CRITICAL ANALYSIS

### 1. Dataset generalisation (the biggest limitation)
**Problem**: The model was trained on CIC-IDS2017 flow features computed by CICFlowMeter. When live traffic is captured with Scapy and features are extracted in real-time, the statistical distributions differ. The model struggles to correctly classify real attacks it hasn't seen in training.

**Root cause**: CIC-IDS2017 flows were captured in a controlled lab with specific tools, timing, and network conditions. Real-world attacks use different tools, speeds, and patterns. The feature distributions shift.

**Evidence**: Kali Linux attacks (Hydra, Nmap, Slowloris) sent through the live engine were frequently misclassified.

**Mitigation in this project**: The demo simulator replays actual CIC-IDS2017 flows, proving the ML pipeline works correctly when given correctly-distributed data.

**What I'd do differently**: Fine-tune the model on a small set of labelled live captures, or use domain adaptation techniques to bridge the distribution gap.

### 2. Class imbalance in training data
**Problem**: Web Attack - XSS (652 flows), Brute Force (1507), Heartbleed (11), Infiltration (36) are severely underrepresented compared to BENIGN (1.9M) and DDoS (128K).

**Impact**: XSS accuracy = 53.3%, Brute Force = 76.7%.

**Mitigation**: SMOTE helps but can't create truly novel patterns from 11 Heartbleed samples. The minority classes don't have enough diversity for the model to learn robust decision boundaries.

**What I'd do differently**: Supplement with additional datasets (CSE-CIC-IDS2018, UNSW-NB15), collect more labelled samples for weak classes, or use cost-sensitive learning.

### 3. Accuracy metric discrepancy
**README states 99.89%** (test set accuracy) vs **randomised evaluation shows 91.9%**.

**Explanation**: 99.89% is the standard test set accuracy where BENIGN flows dominate. Since BENIGN is easy to classify and represents ~90% of test data, overall accuracy is inflated. The randomised evaluation samples equally across all 16 classes (10 per class), giving each class equal weight. 91.9% is the more honest number for "how well does it detect each attack type."

### 4. Single-machine limitation
The engine captures packets on one network interface. It cannot monitor distributed networks or handle traffic volumes exceeding one machine's capacity.

### 5. Bulk features set to zero
6 of 84 features (bulk rate averages) are hardcoded to 0.0 because they require session-level tracking not implemented in real-time. The model was trained with these features, so zeroing them may affect predictions for attack types where bulk behaviour is discriminative.

### 6. Prediction latency
Single prediction takes ~2.3 seconds due to SHAP computation. This is acceptable for flow-level detection (flows last seconds to minutes) but too slow for packet-level real-time IDS. Batch mode mitigates this (4.57ms per flow).

---

## 20. FUTURE WORK

1. **Dataset diversity**: Train on multiple datasets (CIC-IDS2017 + CSE-CIC-IDS2018 + UNSW-NB15) to improve generalisation
2. **Transfer learning / domain adaptation**: Fine-tune on small labelled live traffic samples to bridge the distribution gap
3. **Online learning**: Incrementally update the model on new labelled data without full retraining
4. **Deep learning**: Test LSTM/CNN architectures that may capture sequential packet patterns better than flow-level features
5. **Encrypted traffic**: Add TLS fingerprinting (JA3/JA4) since most modern traffic is encrypted
6. **Distributed capture**: Integrate with Zeek or Suricata for high-throughput multi-interface capture
7. **SIEM integration**: Export alerts to Splunk/ELK for correlation with other security events
8. **Automated retraining pipeline**: Periodic retraining on newly labelled data with model versioning

---

## 21. VIVA QUESTIONS & ANSWERS

### Project Motivation & Scope

**Q: Why did you choose this project?**
Network intrusion detection is a critical cybersecurity challenge. Traditional signature-based IDS (like Snort) can only detect known attacks. ML-based approaches can potentially detect novel attacks by learning behavioural patterns. I wanted to build a complete system -- not just train a model, but deploy it as a working real-time detection pipeline.

**Q: What is the difference between signature-based and anomaly-based IDS?**
Signature-based IDS matches traffic against known attack patterns (rules/signatures). It's fast and accurate for known attacks but blind to zero-day attacks. Anomaly-based IDS learns what "normal" traffic looks like and flags deviations. ML-based IDS is a form of anomaly-based detection that can also learn specific attack patterns from labelled data.

**Q: Why network-level detection and not host-based?**
Network IDS monitors traffic at the network layer, catching attacks before they reach hosts. Host-based IDS monitors individual machines. NIDS provides a single monitoring point for an entire network segment, which is more scalable. My project focuses on network flow classification.

**Q: What is the scope of your project?**
A complete ML pipeline from raw data to real-time prediction: data preprocessing, feature engineering, model training with explainability, a REST API prediction service, a real-time packet capture engine, monitoring dashboards, alerting, and containerised deployment. It classifies network flows into 15 attack types plus benign.

---

### Dataset

**Q: Why CIC-IDS2017 specifically?**
It's the most widely used benchmark dataset for network IDS research. It was created with real network infrastructure (not synthetic), covers a diverse range of attack types, and has ground-truth labels. This makes results comparable with published research.

**Q: What are the limitations of CIC-IDS2017?**
1. Created in 2017 -- doesn't include modern attack types (e.g., ransomware C2, cryptomining)
2. Lab environment -- traffic patterns may not match real-world networks
3. Class imbalance -- some attacks have very few samples (Heartbleed: 11)
4. Feature extraction tool (CICFlowMeter) -- features are tied to this specific tool's computation

**Q: How did you handle the class imbalance?**
Three approaches:
1. SMOTE on the training set to synthesise minority class samples
2. Using f1_macro as the primary metric instead of accuracy
3. The Random Forest uses `class_weight='balanced'` when SMOTE is not applied

**Q: Why not use a more recent dataset?**
CIC-IDS2017 is the standard benchmark. Using it makes my results directly comparable to hundreds of published papers. I acknowledge the limitation and suggest supplementing with CSE-CIC-IDS2018 as future work.

---

### Data Processing

**Q: Why did you remove IP addresses from the features?**
IP addresses are identifiers, not behavioural features. If included, the model would memorise which IPs are attackers in the CIC-IDS2017 lab, and fail completely on any other network. Removing them forces the model to learn attack behaviour patterns instead.

**Q: What is StandardScaler and why use it?**
StandardScaler transforms features to have zero mean and unit variance: `z = (x - mean) / std`. This prevents features with large ranges (e.g., Flow Duration in microseconds) from dominating features with small ranges (e.g., flag counts 0-10). While tree models are technically scale-invariant, scaling ensures consistent behaviour across the ensemble and for SHAP.

**Q: Why apply SMOTE only to the training set?**
If SMOTE were applied before splitting, synthetic samples generated from test data would leak information into training. This would inflate accuracy because the model has "seen" aspects of the test data during training. This is called data leakage and it's a common mistake in ML research.

**Q: What is ANOVA F-test and why use it for feature selection?**
ANOVA (Analysis of Variance) F-test measures the ratio of between-class variance to within-class variance for each feature. A high F-score means the feature's values differ significantly between attack types, making it useful for classification. I selected the top 50 of 84 features, reducing dimensionality while keeping the most discriminative features.

**Q: Why 50 features? Why not 30 or 70?**
50 was chosen as a balance. Fewer features risk losing discriminative information; more features increase noise and computation. 50 retains 60% of features. In practice, I could have used cross-validation to find the optimal K, but 50 performed well empirically.

---

### Models

**Q: Why Random Forest and XGBoost? Why not neural networks?**
1. **Performance on tabular data**: Random Forest and XGBoost consistently outperform neural networks on structured/tabular data (multiple benchmark studies confirm this)
2. **Interpretability**: Tree models are interpretable via feature importance and SHAP. Neural networks are black boxes.
3. **Training efficiency**: Both train in minutes on CPU. Neural networks would require GPU and much more tuning.
4. **No sequence dependency**: Flow features are a fixed vector, not a sequence. Neural networks (LSTM/CNN) are better suited when temporal order within the flow matters.

**Q: Explain how Random Forest works.**
Random Forest is a bagging ensemble. It trains N decision trees independently, each on a random subset of the training data (bootstrap sample) and considering only a random subset of features at each split. The final prediction is the majority vote (classification) or average (regression) across all trees. Randomness reduces overfitting -- individual trees may overfit, but their errors are uncorrelated so they cancel out.

**Q: Explain how XGBoost works.**
XGBoost is a gradient boosting ensemble. It trains trees sequentially: each new tree is trained to correct the residual errors of all previous trees combined. It uses gradient descent on the loss function to determine what the next tree should predict. Key regularisation: L1/L2 penalties on leaf weights, max depth limits, and subsampling. Early stopping halts training when validation loss stops improving.

**Q: What is the difference between bagging and boosting?**
Bagging (Random Forest) trains models independently in parallel on random subsets -- reduces variance. Boosting (XGBoost) trains models sequentially where each corrects the previous -- reduces bias. Bagging is more robust to overfitting; boosting achieves higher accuracy but risks overfitting if not regularised.

**Q: Why ensemble the two models?**
They make different types of errors. RF may miss subtle patterns (higher bias) but is stable (lower variance). XGBoost captures subtleties (lower bias) but may overfit (higher variance). Averaging their probabilities combines their strengths.

**Q: What is soft voting?**
Instead of each model casting a single vote (hard voting), soft voting averages the probability distributions. If RF gives P(DDoS)=0.6 and XGBoost gives P(DDoS)=0.55, the ensemble says P(DDoS)=0.575. This preserves uncertainty information and produces better-calibrated confidence scores.

**Q: What does early stopping do in XGBoost?**
XGBoost monitors validation loss during training. If the validation loss doesn't improve for 10 consecutive rounds (early_stopping_rounds=10), training stops. This prevents overfitting -- the model stops before it starts memorising the training data.

**Q: Why is f1_macro better than accuracy for this problem?**
With ~90% benign traffic, a model predicting everything as BENIGN gets 90% accuracy. F1_macro computes F1 for each class independently and averages them equally. If the model fails on XSS (652 samples), accuracy barely changes but f1_macro drops significantly. F1_macro treats all classes as equally important regardless of their frequency.

---

### Explainability

**Q: What is SHAP and how does it work?**
SHAP (SHapley Additive exPlanations) uses Shapley values from game theory. It treats each feature as a "player" in a "game" (prediction) and calculates each feature's marginal contribution across all possible feature combinations. For tree models, TreeExplainer computes exact Shapley values efficiently by exploiting the tree structure. The output is: "Feature X contributed +0.23 towards class Y."

**Q: Why is explainability important for an IDS?**
Security analysts need to understand why a flow was flagged. A prediction of "DDoS with 95% confidence" is useful, but knowing "Dst Port=80 and high Flow Packets/s were the main indicators" helps analysts validate the alert and respond appropriately. Without explainability, the IDS is a black box that analysts won't trust.

**Q: What's the difference between SHAP and LIME?**
SHAP: Mathematically exact for tree models (TreeExplainer), consistent (features with larger impact always get larger attribution), additive (attributions sum to the prediction).
LIME: Model-agnostic (works with any model), approximates locally by perturbing inputs and fitting a linear model. Faster but less theoretically grounded. Results can vary between runs.

**Q: What are the top features your model relies on?**
Based on SHAP global importance: Dst Port, Flow Duration, Bwd Packet Length Mean, Flow Packets/s, Fwd IAT Max, Packet Length Mean. These make intuitive sense -- port number identifies the service being attacked, duration and packet rates reveal attack intensity, and IAT patterns distinguish automated attacks from human traffic.

---

### Real-Time Engine

**Q: Explain the real-time detection pipeline step by step.**
1. Scapy captures packets on the network interface in a background thread
2. Each packet's metadata (IPs, ports, protocol, flags, lengths) is queued
3. The flow aggregator groups packets into bidirectional flows by 5-tuple
4. When a flow completes (TCP FIN/RST, timeout, or max packets), it's passed to the feature extractor
5. The feature extractor computes 84 CIC-IDS2017 features from the flow's packets
6. The features are sent via HTTP POST to the FastAPI /predict endpoint
7. The API scales features, selects top 50, runs RF + XGBoost ensemble, returns prediction
8. If an attack is detected with sufficient confidence, the alerting system triggers

**Q: Why use HTTP between the engine and the API?**
Decoupling. The engine and API can run on different machines, be scaled independently, and be developed/tested separately. It also means the API can serve other clients (dashboard, external tools) simultaneously. The overhead of HTTP is negligible compared to flow timescales (seconds).

**Q: What happens if the API is down?**
The engine has retry logic (3 attempts with backoff). If all retries fail, the flow is logged as an error and skipped. The engine continues capturing and processing -- it doesn't crash. On startup, it polls `/health` for up to 30 seconds before starting capture.

**Q: Why a 100ms tick rate?**
Balances responsiveness with CPU efficiency. Faster ticks waste CPU on empty checks. Slower ticks delay detection. 100ms means worst-case detection delay is 100ms after a flow completes, which is negligible for network security.

**Q: How does the threading work?**
Main thread handles the capture-aggregate-extract loop. Predictions are submitted to a ThreadPoolExecutor with 4 workers. This prevents slow API calls from blocking packet capture. The pool uses futures -- results are collected asynchronously when ready.

---

### Live Detection Limitation

**Q: Why doesn't the live detection work with real attacks?**
The CIC-IDS2017 features were computed by CICFlowMeter from complete PCAP captures. My real-time feature extractor computes the same 84 features but from live Scapy captures. The statistical distributions differ because:
1. Flow boundaries differ (my timeout is 10s; CICFlowMeter's may be different)
2. Timing precision differs (Scapy timestamps vs PCAP timestamps)
3. The original attacks used specific tools (Ares, LOIC) at specific speeds; Kali tools behave differently
4. Network conditions (latency, packet loss, MTU) differ between the UNB lab and my test environment

**Q: How would you fix this?**
1. **Fine-tuning**: Capture a small labelled dataset of live attacks using my feature extractor, then fine-tune the model on this data
2. **Feature normalisation**: Normalise features relative to baseline traffic rather than using absolute values
3. **Use CICFlowMeter for live capture**: Run CICFlowMeter in real-time instead of my custom extractor, to match the training pipeline exactly
4. **Domain adaptation**: Train a model to map my feature distributions to CIC-IDS2017 distributions

**Q: Is this a common problem in IDS research?**
Yes. It's called the "dataset bias problem" or "distribution shift." Most ML-IDS papers only evaluate on the same dataset they trained on. Very few test on live traffic. This is a widely acknowledged gap between research and deployment.

**Q: Given this limitation, what value does your project provide?**
The project demonstrates a complete, production-grade IDS pipeline. The ML models correctly classify CIC-IDS2017 flows with 91.9% accuracy across all classes. The engineering (API, monitoring, alerting, Docker) is production-ready. The live detection limitation is a research problem, not an engineering problem -- and identifying it is itself a valid research contribution.

---

### Performance

**Q: Why is single prediction so slow (2.3 seconds)?**
SHAP computation dominates. TreeExplainer traces through all trees in both models to compute exact Shapley values. With 200 RF trees + 300 XGBoost trees over 50 features, this is computationally expensive. Batch mode amortises this cost, achieving 219 flows/sec.

**Q: How would you improve latency?**
1. Disable SHAP for production (use it only for investigation)
2. Use approximate SHAP (KernelExplainer) instead of exact TreeExplainer
3. Pre-compute SHAP background data at startup
4. Use batch predictions exclusively
5. Deploy on GPU for parallel tree traversal

**Q: Can this handle enterprise-scale traffic?**
Not on a single machine. Enterprise networks see millions of flows per second. This system handles ~219 flows/sec in batch mode. Scaling would require: distributed packet capture (Zeek/Suricata), message queues (Kafka) between capture and classification, multiple API instances behind a load balancer, and Redis for shared state.

---

### Architecture & Design

**Q: Why FastAPI instead of Flask for the API?**
FastAPI is async-native (Uvicorn ASGI), generates Swagger documentation automatically, validates inputs with Pydantic, and benchmarks 2-3x faster than Flask for API workloads. Flask is used for the dashboard where these features matter less.

**Q: Why separate the API from the engine?**
Separation of concerns. The API is a stateless prediction service that can serve any client. The engine is a stateful packet processing pipeline. Separating them means: the API can be tested independently (with curl/Postman), multiple engines can share one API, the API can be deployed in Docker while the engine runs on bare metal (for network access).

**Q: Why YAML for configuration?**
YAML is human-readable, supports comments (JSON doesn't), and handles nested structures naturally. The config file has hierarchical sections (network, flow, api, alerting) that map cleanly to YAML's indentation structure.

**Q: Why Docker?**
Reproducibility. Docker ensures the API runs identically on any machine regardless of installed Python version, OS, or library versions. Docker Compose orchestrates the full stack (API + Prometheus + Grafana) with a single command. Resource limits prevent the API from consuming all system resources in production.

**Q: Why Prometheus + Grafana and not just logging?**
Logs tell you what happened. Metrics tell you what's happening right now. Prometheus captures time-series metrics (prediction rate, latency, error rate) that can be visualised as graphs, queried historically, and used for alerting. Grafana provides dashboards that operations teams are familiar with. This is the industry standard for production monitoring.

---

### Security

**Q: What security measures does your system have?**
1. API authentication (optional API key with constant-time comparison)
2. Rate limiting (prevents DoS against the API itself)
3. Input validation (rejects malformed requests)
4. Non-root Docker container
5. Read-only model volume mounts
6. No IP addresses stored in model (prevents information leakage)
7. Environment variables for secrets (not hardcoded)
8. CORS configuration (restricts browser access)

**Q: Could an attacker evade your IDS?**
Yes. Evasion techniques include:
1. **Adversarial ML**: Craft network flows that look benign to the model but carry attack payloads
2. **Mimicry attacks**: Modify attack traffic to match benign flow statistics
3. **Fragmentation**: Split attacks across many small flows below detection thresholds
4. **Encryption**: TLS-encrypted traffic hides payload content (my model uses flow metadata, not payload, so it's partially resilient)
5. **Slow-rate attacks**: Attacks below the flow rate thresholds

---

### Methodology & Process

**Q: What development methodology did you follow?**
Iterative development. I built the ML pipeline first (data -> model -> evaluation), then the API, then the real-time engine, then monitoring/alerting. Each component was tested independently before integration. I used git for version control.

**Q: How did you test your system?**
Four levels:
1. **Unit**: API test client validates each endpoint
2. **Integration**: test_attack_detection.py sends real attack flows through the full pipeline
3. **Statistical**: randomised_evaluation.py runs multiple evaluation rounds with different seeds
4. **Performance**: benchmark_suite.py measures latency and throughput

**Q: What was the most challenging part?**
The real-time feature extraction. Matching CIC-IDS2017's 84 features from live packet captures required understanding the exact computation for each feature, handling edge cases (single-packet flows, ICMP, timeouts), and ensuring numerical stability (division by zero, empty lists). The live detection limitation was discovered only after this work was complete.

**Q: What would you do differently if starting over?**
1. Start with CICFlowMeter for feature extraction instead of building a custom one -- this would eliminate the distribution shift problem
2. Use multiple datasets from the start (CIC-IDS2017 + UNSW-NB15) for better generalisation
3. Implement online learning so the model improves from operational feedback
4. Spend less time on the monitoring stack and more on evaluating real-world detection

---

### Theoretical

**Q: What is a confusion matrix and what does yours show?**
A confusion matrix shows predicted vs actual classes. Each row is an actual class, each column is a predicted class. Diagonal entries are correct predictions. Off-diagonal entries show which classes get confused with each other. My confusion matrix shows Web Attack - XSS is frequently confused with Web Attack - Brute Force, and Infiltration is sometimes classified as BENIGN.

**Q: What is precision vs recall?**
Precision = TP / (TP + FP) -- "Of all flows predicted as DDoS, how many actually were?"
Recall = TP / (TP + FN) -- "Of all actual DDoS flows, how many did we catch?"
For an IDS, recall is arguably more important (missing an attack is worse than a false alarm).

**Q: What is F1 score?**
F1 = 2 * (precision * recall) / (precision + recall). It's the harmonic mean of precision and recall. It penalises models that sacrifice one for the other. F1_macro averages F1 across all classes equally.

**Q: What is overfitting and how did you prevent it?**
Overfitting is when a model memorises training data instead of learning general patterns. Prevention:
1. Train/validation/test split (model never sees test data during training)
2. SMOTE applied only to training set (no data leakage)
3. Random Forest's random feature selection at each split
4. XGBoost's L1/L2 regularisation and early stopping
5. Cross-validation to verify consistency across folds

**Q: What is the bias-variance tradeoff?**
Bias: error from oversimplified assumptions (underfitting). Variance: error from sensitivity to training data (overfitting). RF reduces variance (averaging many trees). XGBoost reduces bias (iteratively correcting errors). The ensemble balances both.

**Q: Explain cross-validation.**
K-fold cross-validation splits training data into K equal parts. The model trains on K-1 parts and validates on the remaining part, rotating K times. This gives K accuracy estimates, revealing whether performance is consistent or dependent on the specific data split. I used 3-fold stratified cross-validation (stratified = maintains class proportions in each fold).

---

### Quick Reference Numbers

| Metric | Value |
|--------|-------|
| Dataset size | 2.1M flows |
| Total features | 84 |
| Selected features | 50 |
| Attack classes | 15 + BENIGN = 16 |
| Train/Val/Test split | 70/15/15 |
| RF trees | 200, max_depth=20 |
| XGBoost rounds | 300 (with early stopping at 10) |
| Test accuracy (XGBoost) | 99.44% |
| Test F1 macro (XGBoost) | 0.8337 |
| Randomised eval accuracy | 91.9% +/- 1.2% |
| Weakest class (XSS) | 53.3% |
| Single prediction latency | ~2.3 seconds |
| Batch throughput | 219 flows/sec |
| Batch speedup | 98.3x |
| Flow timeout | 10 seconds |
| Packet queue | 50,000 |
| Max concurrent flows | 100,000 |
| Prediction workers | 4 threads |
| SHAP features returned | Top 5 |
| Email cooldown | 5 minutes |
| Slack cooldown | 2 minutes |
