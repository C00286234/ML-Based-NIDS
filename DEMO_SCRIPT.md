# NIDS Demo Script — FYP Presentation (May 12, 2026)

**Total time:** ~12-15 minutes
**Setup before you start:** Close all unnecessary apps. Have two terminal windows and Chrome ready.

---

## PRE-DEMO SETUP (do this 5 minutes before)

```powershell
# Terminal 1 — Start the API
cd C:\Users\folosiye\ML-BASED-NIDS
python run_nids.py --api-only
# Wait until you see "ready!" in the terminal
```

```powershell
# Terminal 2 — Start the dashboard
cd C:\Users\folosiye\ML-BASED-NIDS\dashboard
python app.py
```

- Open Chrome to **http://localhost:5001** — verify "Connected" badge is green
- Open a second Chrome tab to **http://localhost:8000/docs** (Swagger UI)
- Keep a third terminal ready for the engine command

---

## ACT 1: INTRODUCTION (2 minutes)

> "This is an ML-based Network Intrusion Detection System. It classifies network traffic flows into 15 attack types plus benign using a Random Forest and XGBoost ensemble trained on the CIC-IDS2017 dataset."

**Show the dashboard** (Chrome tab 1) — point out:
- The shield icon and "Connected" status — the system is live
- The four stat cards — currently at zero, about to change
- The empty timeline and detection feed — waiting for traffic

> "The system has three main components: a FastAPI prediction service that runs the ML models, a real-time engine that processes network flows, and this monitoring dashboard connected via Server-Sent Events."

---

## ACT 2: START THE ENGINE — LIVE DETECTION (4-5 minutes)

This is the centrepiece. Start the engine in replay mode:

```powershell
# Terminal 3 — Start engine in replay mode
cd C:\Users\folosiye\ML-BASED-NIDS
python run_nids.py --engine-only --replay data/processed/cicids2017_cleaned.csv --speed 2.0 --attack-ratio 0.4
```

**Switch to the dashboard immediately.** Talk while detections stream in:

> "I'm now replaying real CIC-IDS2017 network flow data through the engine. The engine reads actual feature vectors from the dataset — the same data the model was trained on — and submits them to the API for classification."

**As cards appear, point out:**
- Attack cards flash red/orange when they arrive (DoS GoldenEye, Portscan, FTP-Patator, etc.)
- Benign cards are dimmed — the system correctly identifies normal traffic
- The confidence percentages (most attacks at 99-100%)
- The severity badges (HIGH, MEDIUM, LOW)
- The timeline chart filling with green (benign) and red (attack) data points

> "You can see the system is detecting multiple attack types in real-time — DoS attacks, port scans, brute force attempts — each classified with high confidence."

**Point out the sidebar:**
- Attack Distribution doughnut — shows the mix of attacks detected
- Severity Breakdown — bars filling up for high/medium
- Top Talkers — source IPs generating the most traffic

---

## ACT 3: EXPLAINABILITY — SHAP (2-3 minutes)

**Click on a DoS GoldenEye card** to expand it.

> "Every prediction comes with an explanation. This uses SHAP — SHapley Additive exPlanations — based on game theory. It shows which features contributed most to this classification."

**Point out the expanded detail:**
- **SHAP bars:** "Bwd IAT Std and Bwd Packet Length Std are the strongest indicators — these measure the variability in backward packet timing and size, which is characteristic of DoS GoldenEye's attack pattern."
- **Class probabilities:** "The model assigns 100% probability to DoS GoldenEye. The next most likely classes have near-zero probability."

> "This explainability is critical for security analysts. They don't just get an alert — they understand WHY the system flagged this traffic, so they can validate the detection and respond appropriately."

**Click on a Portscan or FTP-Patator card** to show different SHAP patterns:

> "Notice how different attack types have different contributing features. Port scans are driven by destination port diversity, while brute force attacks show patterns in packet timing."

---

## ACT 4: THE API (2 minutes)

**Switch to the Swagger UI tab** (http://localhost:8000/docs).

> "The classification runs on a FastAPI service. Let me show the API."

**Point out key endpoints:**
- `/predict` — single flow classification with SHAP
- `/batch_predict` — up to 1000 flows at once, 219 flows per second
- `/health` — model status, memory usage, uptime
- `/stats` — prediction counts, attack rate, latency metrics
- `/classes` — the 16 classes the model recognises

**Click on `/health`, hit "Try it out" → "Execute":**

> "The health endpoint shows both models are loaded, the system uptime, and memory usage. This is what the dashboard polls to show the status bar."

---

## ACT 5: ML PIPELINE OVERVIEW (2 minutes)

> "The ML pipeline starts with the CIC-IDS2017 dataset — 2.1 million labelled network flows across 15 attack types."

Key points to mention:
- **Preprocessing:** Cleaned NaN/infinity values, removed IP addresses to prevent memorisation
- **Feature engineering:** StandardScaler normalisation, then SelectKBest reduces 84 features to the top 50 using ANOVA F-test
- **Class imbalance:** SMOTE oversampling on training data only — critical to avoid data leakage
- **Models:** Random Forest (200 trees) + XGBoost (300 rounds with early stopping), combined via soft voting
- **Results:** 99.4% accuracy, but more honestly 93.5% when evaluating equally across all classes (F1 macro)

> "The honest number is 93.5%. Accuracy is inflated because 90% of the data is benign. The model performs well on major attack types but struggles with rare ones like XSS which had only 652 training samples."

---

## ACT 6: LIMITATIONS & HONEST ANALYSIS (1-2 minutes)

> "I want to be transparent about a key limitation. The model was trained on features computed by CICFlowMeter in a controlled lab. When we capture live traffic with Scapy and extract features in real-time, the statistical distributions differ. This is called distribution shift — it's a well-documented challenge in IDS research."

> "That's why this demo uses replay mode — it feeds the actual CIC-IDS2017 feature vectors through the engine, proving the ML pipeline classifies correctly within the engine architecture. The engineering is production-ready; the gap is a research problem that would be solved by retraining on traffic from the target network."

> "This limitation is not unique to my system. Most ML-IDS papers only evaluate on the same dataset they trained on. Identifying and honestly documenting this gap is itself a contribution."

---

## QUESTIONS PREP — LIKELY FOLLOW-UPS

| Question | Quick Answer |
|----------|-------------|
| "Why not neural networks?" | Tree ensembles outperform NNs on tabular data, and they're interpretable via SHAP |
| "Why two models?" | They make different errors — RF reduces variance, XGBoost reduces bias. Soft voting combines strengths |
| "How would you fix the live detection?" | Retrain on a small set of labelled live captures from the target network |
| "Can this scale?" | Current: 219 flows/sec batch. Enterprise: would need Zeek/Suricata + Kafka + multiple API instances |
| "Why CIC-IDS2017?" | Standard benchmark, 600+ citations, makes results comparable to published research |
| "What about encrypted traffic?" | Model uses flow metadata (timing, sizes, ports), not payload. Partially resilient. Future: add JA3/JA4 TLS fingerprinting |

---

## SHUTDOWN (after demo)

```powershell
# Ctrl+C in Terminal 3 (engine)
# Ctrl+C in Terminal 1 (API)
# Ctrl+C in Terminal 2 (dashboard)
```

---

## TIPS

- **Speed:** `--speed 2.0` gives ~2 detections per second. Increase to 4.0 if the demo feels slow, decrease to 1.0 if you want more time to talk between detections.
- **Attack ratio:** `--attack-ratio 0.4` means 40% of replayed flows are attacks. Increase to 0.6 for a more dramatic demo.
- **If something crashes:** The API and dashboard are independent. Restart whichever failed. The dashboard reconnects automatically.
- **Rehearse once** the day before. Run through the full script to check timing and that all commands work.
