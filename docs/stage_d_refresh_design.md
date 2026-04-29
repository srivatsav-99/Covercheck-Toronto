# Stage D — Operational Refresh Design

## Goal

Stage D adds an operational refresh layer to Toronto CoverCheck.

The purpose is to refresh prediction artifacts without manually opening notebooks.

The current system already has:

- processed feature artifacts
- FastAPI serving layer
- Streamlit dashboard
- Docker Compose deployment
- GitHub Actions CI

Stage D focuses on scheduled scoring and artifact refresh.

## Scope

### In Scope

- Load existing engineered feature artifact
- Load final trained model artifacts
- Generate updated prediction artifacts
- Refresh FastAPI/dashboard outputs
- Wrap scoring in a CLI command
- Wrap scoring in a Prefect flow
- Add tests for scoring imports and output schema

### Out of Scope For Now

- Full raw data ingestion
- Live weather API refresh
- Live Ontario 511 refresh
- Full retraining
- Cloud deployment
- Model monitoring dashboard

## Current Artifact Flow

```text
data/processed/features_v2.parquet
        ↓
FastAPI reads existing prediction artifacts
        ↓
Streamlit dashboard renders current forecasts
```

## Target Artifact Flow

```text
data/processed/features_v2.parquet
        ↓
src/scoring/score_latest.py
        ↓
data/processed/surge_predictions.parquet
data/processed/nbhd_predictions.parquet
        ↓
FastAPI
        ↓
Streamlit dashboard
```

## Required Inputs

```text
data/processed/features_v2.parquet
```

## Required Outputs

```text
data/processed/surge_predictions.parquet
data/processed/nbhd_predictions.parquet
```

## Expected Surge Prediction Schema

```text
date
surge_proba_t1
surge_label_t1
surge_proba_t2
surge_label_t2
```

## Expected Neighbourhood Prediction Schema

```text
date
nbhd_id
target_collision_t1
target_ksi_t1
proba_collision_t1
proba_ksi_t1
risk_score_t1
target_collision_t2
target_ksi_t2
proba_collision_t2
proba_ksi_t2
risk_score_t2
```