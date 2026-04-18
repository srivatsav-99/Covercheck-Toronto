# \# 🚀 Toronto CoverCheck

# \## 🛡️ Spatio-Temporal Collision Risk Forecasting System

# 

# Predicts citywide accident surges and identifies high-risk neighbourhoods in Toronto using machine learning, real-world data, and production-grade engineering.

# 

# \---

# 

# \## 🔥 Why This Project Matters

# 

# Road accidents are not random.

# 

# They are driven by:

# 

# \- Weather conditions 🌧️  

# \- Traffic disruptions 🚧  

# \- Historical patterns 📊  

# 

# This system answers:

# 

# > \*\*“Where are accidents most likely to happen tomorrow?”\*\*

# 

# \---

# 

# \## 🧠 What This System Does

# 

# \### 1. Citywide Risk Forecast

# \- Predicts probability of collision surge (T+1, T+2)

# \- Helps identify high-risk days

# 

# \### 2. Neighbourhood Risk Ranking

# \- Ranks all 158 Toronto neighbourhoods

# \- Identifies Top-K high-risk zones

# 

# \### 3. Interactive Risk Dashboard

# \- Visual map (Folium)

# \- Risk distribution

# \- Zone breakdown

# 

# \---

# 

# \## 🏗️ System Architecture

Raw Data

↓

Feature Engineering

↓

ML Models

↓

Parquet Artifacts

↓

FastAPI

↓

Streamlit Dashboard





\---



\## 📊 Data Sources



\- Toronto Police Collision Data  

\- KSI (Killed \& Seriously Injured)  

\- Ontario 511 Traffic Disruptions  

\- Weather Data (Open-Meteo)  

\- Toronto Neighbourhood Boundaries  



\---



\## ⚙️ Machine Learning Pipeline



\### 🔹 Citywide Model

\- LightGBM

\- Predicts surge probability (binary classification)

\- Time-aware validation (no leakage)



\### 🔹 Neighbourhood Model

\- Predicts per-neighbourhood collision risk

\- Incorporates:

&#x20; - Local features

&#x20; - Global “surge tide” signal



\---



\## 🧪 Model Evaluation



\- ROC-AUC  

\- PR-AUC  

\- Brier Score  

\- Precision@K (business metric)  



\---



\## 🖥️ Dashboard Preview



\### Risk Map

\- Choropleth visualization

\- Top zones highlighted



\### Top Risk Zones

\- Ranked table

\- Collision probability + expected collisions



\### Zone Distribution

\- High / Medium / Low segmentation



\---



\## 🚀 API (FastAPI)



Available at:

http://localhost:8000/docs





\### Endpoints



\- `/health` → system status  

\- `/surge/latest` → citywide risk  

\- `/neighbourhoods/topk` → top risk zones  

\- `/metrics` → model metrics  



\---



\## 🐳 Run the Project (Docker)



\### 1. Clone repo



```bash

git clone https://github.com/<your-username>/Covercheck-Toronto.git

cd Covercheck-Toronto



\### 2. Run containers



docker compose up --build



\### 3. Access



Dashboard → http://localhost:8501

API → http://localhost:8000/docs



\---



\## 🧪 Run Tests



pytest -q



\---



\## ⚡ CI Pipeline



\### GitHub Actions

Automated:



Tests

Dependency validation

Import checks



\---



\## 🧩 Tech Stack



\### ML \& Data



\- Python

\- pandas / numpy

\- LightGBM



\### Geospatial



\- GeoPandas

\- Folium



\### Backend



\- FastAPI



Frontend

\- Streamlit



\### DevOps



\- Docker

\- GitHub Actions (CI)



\---



\## ⚠️ Limitations



\- Static data (no real-time ingestion yet)

\- No scheduled retraining

\- Single-machine deployment



\---



\## 🚀 Future Work



\- Real-time data ingestion

\- Automated daily refresh (Prefect)

\- Azure deployment

\- Model monitoring



\---



\## 👨‍💻 Author



Srivatsav Shrikanth

Machine Learning \& Data Analytics



