from fastapi import FastAPI, HTTPException, Query

from src.api.models import (
    HealthResponse,
    SurgeLatestResponse,
    TopKResponse,
    MetricsResponse,
)
from src.api.services import (
    get_latest_surge,
    get_topk_neighbourhoods,
    get_metrics,
)

app = FastAPI(
    title="CoverCheck API",
    description="Serving layer for Toronto CoverCheck artifacts",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "api": "CoverCheck API",
        "version": "1.0.0",
    }


@app.get("/surge/latest", response_model=SurgeLatestResponse)
def surge_latest():
    try:
        return get_latest_surge()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/neighbourhoods/topk", response_model=TopKResponse)
def neighbourhoods_topk(
    horizon: int = Query(1, ge=1, le=2),
    k: int = Query(10, ge=1, le=50),
):
    try:
        return get_topk_neighbourhoods(horizon=horizon, k=k)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    try:
        return get_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))