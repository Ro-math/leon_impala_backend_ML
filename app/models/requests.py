from typing import List, Optional
from pydantic import BaseModel
from app.core.entities import ImpalaAction, LionAction

class TrainingStartRequest(BaseModel):
    num_incursions: int
    initial_positions: List[int]
    impala_mode: str # "random" or "programmed"
    impala_sequence: Optional[List[ImpalaAction]] = None

class HuntingStartRequest(BaseModel):
    lion_position: int # 1-8
    impala_mode: str
    impala_sequence: Optional[List[ImpalaAction]] = None

class HuntingStepRequest(BaseModel):
    # If manual control is needed? Or just trigger next step?
    # Prompt says "Cacería paso a paso... realizar una incursión... seguida paso a paso".
    # Usually step just advances time.
    pass

class KnowledgeSaveRequest(BaseModel):
    filename: str
    format: str = "json"

class KnowledgeLoadRequest(BaseModel):
    filename: str

class HuntingExplainRequest(BaseModel):
    time_step: Optional[int] = None
