from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    api: str
    version: str


class SurgeLatestResponse(BaseModel):
    date: str
    surge_proba_t1: Optional[float] = None
    surge_proba_t2: Optional[float] = None


class NeighbourhoodRiskRecord(BaseModel):
    date: str
    nbhd_id: int
    area_name: Optional[str] = None
    collision_prob_t1: Optional[float] = None
    collision_prob_t2: Optional[float] = None
    rank_t1: Optional[int] = None
    rank_t2: Optional[int] = None


class TopKResponse(BaseModel):
    horizon: int
    k: int
    as_of_date: str
    records: List[NeighbourhoodRiskRecord]


class MetricsResponse(BaseModel):
    surge_metrics: List[Dict[str, Any]]
    nbhd_metrics: List[Dict[str, Any]]