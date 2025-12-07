from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from app.core.entities import Lion, Impala

class SimulationStateResponse(BaseModel):
    lion: Lion
    impala: Impala
    time_step: int
    status: str

class TrainingStatusResponse(BaseModel):
    status: str # "running", "completed", "stopped"
    progress: float
    current_incursion: int
    total_incursions: int
    success_count: int
    fail_count: int

class HuntingStepResponse(BaseModel):
    lion: Lion
    impala: Impala
    time_step: int
    status: str
    impala_action: str
    lion_action: str
    info: str

class KnowledgeResponse(BaseModel):
    q_table_size: int
    abstractions_count: int

class VisualizationMapResponse(BaseModel):
    map_width: int
    map_height: int
    waterhole: List[List[int]] # [[x1, y1], [x2, y2]]
    lion_positions: Dict[int, List[int]] # {1: [x, y], ...}
    impala_position: List[int]

class VisionAreasResponse(BaseModel):
    triangle_points: List[List[float]] # [[x1, y1], [x2, y2], [x3, y3]]

class HistoryResponse(BaseModel):
    history: List[Dict[str, Any]] # List of state dicts

class TrainingStatisticsResponse(BaseModel):
    total_incursions: int
    success_rate: float
    avg_steps: float
    success_rate_by_position: Dict[int, float]
    abstractions_count: int
    q_table_size: int

class HuntingExplainResponse(BaseModel):
    explanation: str
    relevant_rules: List[str]
    q_values: Dict[str, float]

class HuntingResultResponse(BaseModel):
    result: str # "success", "failed", "in_progress"

class KnowledgeFilesResponse(BaseModel):
    files: List[str]

class KnowledgeQueryResponse(BaseModel):
    best_action: str
    q_values: Dict[str, float]
    matching_rules: List[str]
