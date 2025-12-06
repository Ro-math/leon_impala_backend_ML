from fastapi import APIRouter, HTTPException
from app.models.responses import VisualizationMapResponse, VisionAreasResponse, HistoryResponse
from app.core.entities import GameMap, ImpalaAction
from app.core.vision_calculator import VisionCalculator
from app.api.hunting import current_hunt_state # Access current hunt state

router = APIRouter()

@router.get("/map", response_model=VisualizationMapResponse)
def get_map():
    gm = GameMap()
    return VisualizationMapResponse(
        map_width=gm.width,
        map_height=gm.height,
        waterhole=[list(gm.waterhole_top_left), list(gm.waterhole_bottom_right)],
        lion_positions={k: list(v) for k, v in gm.valid_lion_positions.items()},
        impala_position=[9, 9]
    )

@router.get("/vision-areas", response_model=VisionAreasResponse)
def get_vision_areas(direction: str):
    vc = VisionCalculator()
    points = []
    
    if direction == "front":
        points = [list(vc.p8), list(vc.I), list(vc.p2)]
    elif direction == "left":
        points = [list(vc.p8), list(vc.I), list(vc.p6)]
    elif direction == "right":
        points = [list(vc.p2), list(vc.I), list(vc.p4)]
    else:
        raise HTTPException(status_code=400, detail="Invalid direction. Use 'front', 'left', or 'right'.")
        
    return VisionAreasResponse(triangle_points=points)

@router.get("/history", response_model=HistoryResponse)
def get_history():
    # We need to store history in GameState
    # Current implementation of GameState has self.history = [] but it's not populated in GameEngine.step yet.
    # We need to update GameEngine to populate history.
    
    if current_hunt_state is None:
        return HistoryResponse(history=[])
        
    # Assuming GameEngine populates history now (need to update it)
    return HistoryResponse(history=current_hunt_state.history)
