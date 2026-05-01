# 🚀 Toronto CoverCheck
## 🛡️ End-to-End Collision Risk Forecasting System (ML + MLOps + Deployment)

A production-style machine learning system that predicts citywide collision surges and ranks high-risk neighbourhoods in Toronto, built with full MLOps lifecycle support including training, model governance, monitoring, and deployment.

---

## 🔥 Why This Project Exists

Road accidents are not random — they are driven by:

- Weather conditions 🌧️  
- Traffic disruptions 🚧  
- Temporal patterns 📊  

This system answers a real-world question:

> **“Where and when are collisions most likely to happen tomorrow?”**

---

## 🎯 Real-World Impact

This system enables **proactive decision-making** for:

- 🚓 Emergency services → resource allocation  
- 🚧 Traffic management → disruption planning  
- 🏢 Insurance companies → risk modeling  
- 🏙️ City planners → safer infrastructure  

👉 Moves from **reactive response → predictive risk management**

---

## 🧠 System Capabilities

### 1. Citywide Risk Forecast
- Predicts probability of a **collision surge (T+1, T+2)**
- Identifies high-risk days before they occur

### 2. Neighbourhood Risk Ranking
- Scores all **158 Toronto neighbourhoods**
- Produces **Top-K high-risk zones**

### 3. Interactive Dashboard
- Geospatial risk map (Folium)
- Risk segmentation (High / Medium / Low)
- Trend + performance insights

---

## 🏗️ End-to-End ML Architecture

Raw Data
   ↓
Feature Engineering
   ↓
Training Pipeline (MLflow)
   ↓
Model Registry (Champion / Candidate)
   ↓
Promotion Logic (Metric Guardrails)
   ↓
Scoring Pipeline
   ↓
Monitoring (Drift + Freshness)
   ↓
FastAPI (Serving Layer)
   ↓
Streamlit Dashboard

---

## ⚙️ MLOps System (Core Highlight)

This project includes a full ML lifecycle system, not just modeling.

### 1. Training Pipeline
- Automated candidate model training
- MLflow experiment tracking
- Metric logging (ROC-AUC, PR-AUC, Precision@K)

### 2. Model Registry
- Structured registry:
```text
models/registry/
    citywide/champion
    neighbourhood_collision/champion
    candidates/<timestamp>
```

- Tracks:
-- model artifacts
-- metadata (metrics, features, version)

### 3. Safe Model Promotion

- Candidate replaces production model ONLY IF it outperforms champion
```text
"precision_at_10": 0.815
"champion_precision_at_10": 0.90
"promoted": false
```
👉 Prevents bad models from reaching production

### 4. Scoring Pipeline

- Uses champion model only
- Generates:
-- citywide predictions
-- neighbourhood predictions

### 5. Monitoring System

Tracks:

#### 📊 Data Drift
- Feature distribution monitoring

#### 📉 Prediction Drift
- Changes in model output distribution

⏱️ Freshness
```text
"days_since_update": 122
```

👉 Detects stale models automatically

---

## 🚀 Pipeline Orchestration

Run full system:

```bash
python -m src.pipeline.run_pipeline
```

Executes:
- Train candidate
- Evaluate & promote
- Score latest
- Monitor system

---

## 📊 Data Sources

- Toronto Police Collision Data  
- KSI (Killed & Seriously Injured)  
- Ontario 511 Traffic Disruptions  
- Weather Data (Open-Meteo)  
- Toronto Neighbourhood Boundaries  

---

## ⚙️ Machine Learning Pipeline

### 🔹 Citywide Model
- LightGBM classifier  
- Predicts **surge probability**  
- Time-aware validation (no leakage)

### 🔹 Neighbourhood Model
- Predicts **collision risk per neighbourhood**
- Combines:
  - Local features  
  - Global “surge signal”

---

## 📈 Model Performance

| Metric | Value |
|------|------|
| ROC-AUC | ~0.72 |
| PR-AUC | ~0.75 |
| Precision@10 | ~0.90 |
| Brier Score | ~0.21 |

👉 High Precision@K = strong real-world usefulness

---

## 🖥️ Dashboard Features

- 📍 Interactive risk map (GeoSpatial)
- 📊 Top risk zones ranking
- 📉 Trend & seasonality analysis
- 📦 Model performance tracking

---

## 🚀 Deployment Architecture

- 🐳 Dockerized services (API + Dashboard)
- ⚡ FastAPI serving layer
- 🎛️ Streamlit frontend
- ☁️ Azure Container Apps deployment
- 📦 Slim artifact optimization for performance

---

## 📸 Screenshots

- Risk Map  
- Trends  
- Model Performance  
- API Swagger UI  

*(See /assets folder)*

---

## 🔌 API (FastAPI)

Endpoints:

- `/health` → system status  
- `/surge/latest` → citywide forecast  
- `/neighbourhoods/topk` → risk zones  
- `/metrics` → model performance  

---

## 🐳 Run Locally

```bash
git clone https://github.com/<your-username>/Covercheck-Toronto.git
cd Covercheck-Toronto
docker compose up --build
```

### Access
- Dashboard → http://localhost:8501
- API → http://localhost:8000/docs

---

## 🧩 Tech Stack

### ML & Data
- Python, pandas, numpy
- LightGBM
- MLflow

### Geospatial
- GeoPandas, Folium

### Backend
- FastAPI

### Frontend
- Streamlit

### DevOps
- Docker
- Azure Container Apps
- GitHub Actions
- Custom Model Registry
- Monitoring Pipeline

---

## 🧠 What Makes This Different

Most ML projects stop at modeling.

This system includes:

✅ End-to-end pipeline
✅ Model governance (safe promotion)
✅ Experiment tracking (MLflow)
✅ Monitoring (drift + freshness)
✅ Automated orchestration
✅ Cloud deployment

👉 Built like a production ML system with guardrails

---

## ⚠️ Current Limitations

- Uses local MLflow tracking (no remote server yet)
- Data versioning not implemented (DVC planned)
- Batch pipeline (no real-time streaming)
- Single-region deployment (Azure Container Apps)

--- 

## 🚀 Next Phase (Planned Enhancements)

- MLflow Model Registry (production-grade)
- Data Versioning (DVC / LakeFS)
- Real-time ingestion (Kafka / Event Hub)
- Monitoring dashboards (Grafana)
- Multi-region deployment

---

## 👨‍💻 Author

- Srivatsav Shrikanth
- Machine Learning | Data Analytics | MLOps
