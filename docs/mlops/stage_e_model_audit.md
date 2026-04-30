\# Stage E — Model Audit and MLOps Upgrade Plan



\## Current Status



Toronto CoverCheck currently has:



\- FastAPI serving layer

\- Streamlit dashboard

\- Dockerized API and dashboard containers

\- Azure Container Apps deployment

\- GitHub Actions CI

\- Prefect scoring flow

\- Local model artifacts used by scoring



\## Current Model Artifact Situation



Current production-like model files:



```text

models/citywide\_model.pkl

models/nbhd\_collision\_model.pkl

