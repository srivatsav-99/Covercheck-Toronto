# 🚀 Toronto CoverCheck
## 🛡️ End-to-End Collision Risk Forecasting System (ML + MLOps + Deployment)

A production-style machine learning system that predicts **citywide collision surges** and identifies **high-risk neighbourhoods in Toronto**, deployed using **FastAPI + Streamlit + Docker + Azure**.

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

## 🧠 What the System Does

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

## 🏗️ System Architecture

Raw Data
↓
Feature Engineering
↓
ML Models (LightGBM)
↓
Parquet Artifacts
↓
FastAPI (Serving Layer)
↓
Streamlit Dashboard (Visualization)


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

---

## 🧠 What Makes This Different

Most ML projects stop at modeling.

This system includes:

✅ End-to-end pipeline
✅ Real-world data integration
✅ Spatio-temporal modeling
✅ API serving layer
✅ Interactive dashboard
✅ Cloud deployment

👉 Built like a production ML system, not a notebook.

---

## ⚠️ Limitations

- No real-time streaming
- No automated retraining yet
- Single-region deployment

--- 

## 🚀 Future Work

- Real-time ingestion (streaming pipelines)
- MLOps (MLflow, monitoring, retraining)
- CI/CD automation
- Multi-region deployment

---

## 👨‍💻 Author

- Srivatsav Shrikanth
- Machine Learning | Data Analytics | MLOps
