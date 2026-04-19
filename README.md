# 🚍 Sivas Intelligent Transit ETA Oracle
*Team-003 Hackathon Submission | Case #2: Predictive Transit*

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42.0-FF4B4B.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-brightgreen.svg)](https://xgboost.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Production_Ready-success.svg)]()

## 📖 Project Overview
Welcome to the **Sivas Intelligent Transit ETA Oracle**. This project provides a production-ready Machine Learning pipeline and an interactive Predictive Kiosk dashboard. We successfully replace naive, static bus schedules with dynamic, physics-based AI predictions that account for real-world friction. 

Our architecture solves two core tasks for next-generation transit systems:
1. **Arrival Time Prediction (ETA)**
2. **Crowd Estimation**

---

## 🧠 Core Architecture (The Two-Pillar AI)

We rely on Gradient Boosted Trees (XGBoost) due to their unparalleled performance on tabular data and excellent capabilities with categorical parameters.

### Model 1: The ETA Oracle
Calculates true arrival times by adding real-time friction parameters to the static schedules. Rather than just relying on generic speed averages, our model ingests:
* **Distance to next stop**
* **Cumulative timetable drift** (cascading delay)
* **Real-time traffic levels**
* **Live weather conditions**

### Model 2: Crowd Estimation & Model Chaining
Predicts `passengers_waiting` at any given stop. 
**Advanced Implementation:** We successfully implemented **Model Chaining**. We feed the predicted ongoing delay (the output from Model 1) directly into Model 2 as the `cumulative_delay_min` feature. This gives the AI the critical signal it requires to understand that a delayed bus inherently accumulates a dramatically larger crowd.

### 🛡️ Zero Data Leakage
Data integrity is critical. We utilized a strict **sequential temporal split**.
* **Train Set:** March 1–23, 2025
* **Test Set:** March 24–30, 2025 (Strictly Out-of-Time)
We ruthlessly dropped post-hoc features (like `is_delayed`, `dwell_time`, `crowding_level`) to guarantee zero data leakage.

---

## 🚀 Performance & Evaluation (State-of-the-Art)

To prove our architecture was structurally sound, we first successfully built and tested our pipeline on **30 million rows** of real Boston MBTA industry-standard data, achieving a world-class MAE of ~3.2 minutes.

We then pointed our generalized pipeline directly to the bespoke **Sivas** localized dataset. 
**Our out-of-time results are near-perfect:**

| Model | Primary Metric (Error) | Context |
|---|---|---|
| **ETA Oracle** | **MAE = ~0.8 min** | (~12 seconds of prediction error). R² = 0.99 |
| **Crowd Estimation** | **RMSE = 6.185 pax** | Evaluado en ventana de test desconectada |

---

## 📂 Project Structure
Our codebase is modular and fully decoupled. Environment parameters (`.env`) handle hot-swapping between datasets (Boston vs Sivas) cleanly.

```text
~/project/
├── ai-engine/                  # The Machine Learning backend
│   ├── .env                    # Environment config (Boston/Sivas switch)
│   ├── config.py               # Global path resolution & dictionary mappings
│   ├── pipelines/
│   │   ├── eta_pipeline/       # Preprocessing, Training, Eval (Model 1)
│   │   └── crowd_pipeline/     # Preprocessing, Training, Eval (Model 2)
│   └── models/                 # Serialized JSON models
│
└── dashboard/                  # The Frontend Kiosk
    └── app.py                  # Streamlit Interactive Dashboard
```

---

## 💻 How to Run (VM Deployment)

To deploy the streamlined presentation kiosk on port `8000`:

```bash
cd ~/project/dashboard
source .venv/bin/activate
nohup streamlit run app.py --server.port=8000 --server.address=0.0.0.0 > streamlit.log 2>&1 &
```

*(You can monitor the dashboard output using `tail -f dashboard/streamlit.log`)*