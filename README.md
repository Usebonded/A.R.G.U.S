# 🛡️ SentinelFraud AI - Fraud Detection System Prototype

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20NetworkX-F7931E.svg)](https://scikit-learn.org/)

An enterprise-grade, end-to-end AI Fraud Detection System prototype featuring **multi-entity data ingestion**, **hybrid pattern recognition** (supervised ML + unsupervised anomaly detection + graph syndicate networks), **calibrated 0–100 risk scoring with Explainable AI (XAI)**, and **continuous model adaptation** via human-in-the-loop active learning.

---

## 🌟 Core Pillars

### 1. 🛰️ Multi-Entity Data Ingestion (`src/schema.py`, `src/synthetic_data.py`)
Gathers and correlates multimodal signals across 5 threat dimensions:
- **Identity Signals**: Account age (days), KYC verification tier (0–3), disposable email risk index, historical dispute count.
- **Payment Method Signals**: Tokenized card identifier, brand, type (Credit, Debit, Prepaid), card country vs. billing country mismatch, amount (USD).
- **Location & Geo-Velocity**: Latitude/Longitude, IP country/city, distance from user billing address (km), elapsed minutes between transactions, physical travel velocity in km/h (flags impossible travel >800 km/h).
- **Network & Device Signals**: Hardware fingerprint hash, IP address, ISP/hosting ASN, anonymizer indicators (VPN / Tor / Proxy exit nodes), device trust score, emulator detection.
- **Session & Behavioral Telemetry**: Session duration, failed logins in last 24h, checkout page duration (sub-3s bot detector), robotic mouse cadence anomaly score.

### 2. 🕸️ Hybrid Pattern Recognition Engine (`src/pattern_recognition.py`)
Spots hidden anomalies and non-linear relationships that traditional static rules miss:
- **Supervised ML Classification (`RandomForestClassifier`)**: Predicts calibrated fraud probabilities \(P(\text{Fraud} \mid X)\) based on historical behavior.
- **Unsupervised Anomaly Detection (`IsolationForest`)**: Flags out-of-distribution, zero-day fraud vectors without requiring labeled data.
- **Entity Graph Syndicate Detector (`NetworkX`)**: Constructs entity graphs linking Users, Devices, Cards, and IPs. Identifies coordinated fraud rings (e.g. 1 device recycled across 3+ separate accounts).

### 3. 🎯 Calibrated Risk Scoring & Explainable AI (`src/risk_engine.py`)
- **Composite Risk Score (0 to 100)**: Calibrated blend of Supervised ML probability (45%), Unsupervised Anomaly score (25%), Graph Syndicate risk (15%), and Rule/Heuristic penalties.
- **Automated Decision Tiers**:
  - `ALLOW` (Score < 30): Frictionless automated clearance for legitimate users.
  - `MANUAL REVIEW` (Score 30–74): Amber alert routed to human analysts for manual review.
  - `BLOCK` (Score ≥ 75): Automated rejection for high-risk attacks.
- **Explainable AI (XAI) Decision Drivers**: Transparent attribution explaining point impacts (e.g. `+35 pts: Impossible Travel Velocity 942 km/h`, `+30 pts: Syndicate Device Recycling across 3 accounts`).
- **5-Dimensional Threat Radar**: Radar visualization showing normalized risk sub-scores across Identity, Payment, Location, Network, and Session.

### 4. 🔄 Continuous Adaptation & Active Learning (`src/continuous_adaptation.py`)
- **Analyst Review Hub (Human-in-the-Loop)**: Queue where analysts inspect flagged transactions (scores 30–74), review evidence, and record decisions (`APPROVE_LEGITIMATE` or `CONFIRM_FRAUD`).
- **Real-Time Concept Drift Monitoring**: Monitors rolling transaction windows to detect distribution shifts (spikes in proxy traffic, velocity surges, headless bot bursts).
- **Dynamic Incremental Retraining**: Ingests analyst-verified ground-truth feedback, updates decision boundaries, mints new model versions (`v1.0` → `v1.1`), and tracks performance trajectories (ROC-AUC, Precision, Recall, F1).

---

## 📁 Project Directory Structure

```text
├── app.py                      # Main Streamlit dashboard application
├── requirements.txt            # Project dependencies
├── README.md                   # Complete usage & architecture guide
├── src/
│   ├── __init__.py
│   ├── schema.py               # Multi-entity transaction schema (Dataclass)
│   ├── synthetic_data.py       # Data generator & live stream simulator
│   ├── pattern_recognition.py  # Feature extractor, Isolation Forest, RF & NetworkX
│   ├── risk_engine.py          # Calibrated 0-100 scoring, tiers & XAI drivers
│   └── continuous_adaptation.py # Feedback queue, drift detector & model retrainer
└── tests/
    └── test_pipeline.py        # Automated test suite
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 installed.

### 2. Environment Setup

Clone or extract the project archive:
```bash
cd "savar pbl"
```

Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```
*(Alternatively, if using `uv`: `uv pip install -r requirements.txt`)*

---

## 🧪 Running the Automated Tests

Verify that all ingestion, pattern recognition, risk scoring, and continuous adaptation pipelines function correctly:

```bash
python tests/test_pipeline.py
```

Expected output:
```text
Testing synthetic data generation...
✓ Synthetic data generation passed.
Testing pattern recognition engine training & inference...
Normal transaction risk score: 0 (ALLOW)
Impossible Travel risk score: 79 (BLOCK)
✓ Pattern recognition & risk scoring passed.
Testing continuous adaptation and dynamic retraining...
Drift report: {'drift_detected': True, ...}
Retrained model version info: {'version': 'v1.1', ...}
✓ Continuous adaptation passed.

ALL TEST SUITES PASSED SUCCESSFULLY!
```

---

## 💻 Launching the Streamlit Frontend

Start the interactive dashboard:
```bash
streamlit run app.py
```

Open your browser to:
👉 **`http://localhost:8501`**

---

## 🖥️ Using the Dashboard

The dashboard provides 5 specialized views accessible via the sidebar:

### 1. 🛰️ Live Feed & Ingestion
- **Live Stream Ingestion**: Use the sidebar to inject specific attack scenarios:
  - *Stochastic Real Traffic (Normal)*
  - *Impossible Travel (Geo-Velocity)*
  - *Credential Stuffing ATO (Account Takeover)*
  - *Card Testing Bot (Micro-probes)*
  - *Syndicate Fraud Ring (Device Recycling)*
- Click **"▶️ Ingest 1 Event"** or **"⏩ Burst (5 Events)"** to stream events in real time.
- **KPI Metrics Bar**: Tracks Total Ingested, Auto-Allowed, Review Queue, Auto-Blocked, and Total Fraud Prevented ($).
- **Custom Transaction Evaluator**: Switch to the *🧪 Custom Transaction Evaluator* tab to configure custom attributes across all 5 dimensions and test scoring on-the-fly.

### 2. 🔍 Transaction Inspector & Scorecard
- Select any transaction from the dropdown to perform forensic analysis.
- **0–100 Gauge Chart**: Shows the composite score with its color-coded action badge (`ALLOW` in green, `MANUAL REVIEW` in amber, `BLOCK` in red).
- **5-Dimensional Threat Surface Radar**: Evaluates Identity, Payment, Location, Network, and Session threat levels.
- **XAI Decision Drivers**: Highlights the exact signals that drove the score up (e.g. `+35 pts: Speed 942 km/h exceeds maximum jet travel capability`).

### 3. 🕸️ Pattern Recognition & Graph Rings
- **Entity Relationship Network**: Interactive Plotly graph visualizing connections between Users (🔵), Devices (🔴), and Cards (🟡). High-degree nodes reveal shared device rings.
- **Isolation Forest Outlier Projection**: 2D scatter plot showing transaction distribution across velocity vs. amount, where bubble size reflects unsupervised anomaly severity.

### 4. 🛡️ Analyst Review Hub (HITL)
- Displays all transactions flagged with scores **30–74**.
- Inspect suspicious indicators, enter analyst notes, and take one-click action:
  - **✅ Approve**: Clears transaction and records ground truth as legitimate.
  - **🛑 Confirm Fraud**: Confirms attack and records ground truth as fraudulent.
- Decisions are saved into the **Analyst Feedback Log** for active learning.

### 5. 🔄 Continuous Adaptation & Drift
- **Concept Drift Monitor**: Displays real-time statistical drift alerts (e.g. surge in anonymizers or impossible velocity checkouts).
- **⚡ Retrain Model with Feedback**: Click the retrain button to trigger online incremental retraining with logged human feedback.
- **Model Evolution Table & Trajectory Chart**: Tracks how accuracy, ROC-AUC, precision, recall, and F1 improve across model versions (`v1.0` → `v1.1` → `v1.2`).

---

## 🛡️ License
MIT License. Built for advanced fraud prevention prototyping.
