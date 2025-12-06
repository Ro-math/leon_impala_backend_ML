from fastapi import APIRouter, HTTPException
from app.models.responses import SimulationStateResponse
from app.core.game_engine import GameState, GameMap

router = APIRouter()

# Global simulation state (for simple simulation mode, separate from training)
# In a real app, this might be per-session or in a database.
current_simulation_state = None

@router.post("/initialize")
def initialize_simulation(lion_pos_idx: int = 1):
    global current_simulation_state
    if lion_pos_idx not in GameMap.valid_lion_positions:
        raise HTTPException(status_code=400, detail="Invalid lion position index")
    
    start_pos = GameMap.valid_lion_positions[lion_pos_idx]
    current_simulation_state = GameState(lion_start_pos=start_pos)
    return {"message": "Simulation initialized"}

@router.get("/state", response_model=SimulationStateResponse)
def get_simulation_state():
    global current_simulation_state
    if current_simulation_state is None:
        raise HTTPException(status_code=404, detail="Simulation not initialized")
    
    return SimulationStateResponse(
        lion=current_simulation_state.lion,
        impala=current_simulation_state.impala,
        time_step=current_simulation_state.time_step,
        status=current_simulation_state.status
    )

@router.post("/reset")
def reset_simulation():
    global current_simulation_state
    current_simulation_state = None
    return {"message": "Simulation reset"}
