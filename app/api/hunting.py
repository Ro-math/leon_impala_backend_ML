from fastapi import APIRouter, HTTPException
from app.models.requests import HuntingStartRequest, HuntingStepRequest, HuntingExplainRequest
from app.models.responses import HuntingStepResponse, HuntingExplainResponse, HuntingResultResponse
from app.core.game_engine import GameEngine, GameState, GameMap, ImpalaAction
from app.core.entities import LionAction
from app.api.training import training_manager # Share the KB/Agent

router = APIRouter()

current_hunt_state = None
current_hunt_request = None

@router.post("/start")
def start_hunting(request: HuntingStartRequest):
    global current_hunt_state, current_hunt_request
    
    if request.lion_position not in GameMap.valid_lion_positions:
        raise HTTPException(status_code=400, detail="Invalid lion position")
        
    start_pos = GameMap.valid_lion_positions[request.lion_position]
    current_hunt_state = GameState(lion_start_pos=start_pos)
    current_hunt_request = request
    
    return {"message": "Hunting started"}

@router.post("/step", response_model=HuntingStepResponse)
def step_hunting():
    global current_hunt_state, current_hunt_request
    if current_hunt_state is None:
        raise HTTPException(status_code=400, detail="Hunting not started")
        
    if current_hunt_state.status != "in_progress":
        raise HTTPException(status_code=400, detail="Hunting already finished")

    # Determine Impala Action
    impala_action = ImpalaAction.LOOK_FRONT
    if current_hunt_request.impala_mode == "random":
        import random
        choices = [a for a in ImpalaAction if a != ImpalaAction.FLEE]
        impala_action = random.choice(choices)
    elif current_hunt_request.impala_mode == "programmed" and current_hunt_request.impala_sequence:
        seq_idx = current_hunt_state.time_step % len(current_hunt_request.impala_sequence)
        impala_action = current_hunt_request.impala_sequence[seq_idx]

    # Lion Decision
    # Use the trained agent
    state_key = training_manager.agent.get_state_key(
        current_hunt_state.lion.position, 
        impala_action, 
        current_hunt_state.lion.state
    )
    lion_action = training_manager.agent.choose_action(state_key)
    
    # Execute Step
    # Note: We need to pass the engine instance
    engine = GameEngine()
    next_state, reward, done, info = engine.step(current_hunt_state, lion_action, impala_action)
    
    current_hunt_state = next_state
    
    # Get the actual actions performed from history (they might have been overridden)
    # Impala action can be overridden to FLEE
    # Lion action can be overridden to ATTACK (attack persistence rule)
    last_step_info = current_hunt_state.history[-1]
    actual_impala_action = last_step_info["impala_action"]
    actual_lion_action = last_step_info["lion_action"]
    
    return HuntingStepResponse(
        lion=current_hunt_state.lion,
        impala=current_hunt_state.impala,
        time_step=current_hunt_state.time_step,
        status=current_hunt_state.status,
        impala_action=actual_impala_action,
        lion_action=actual_lion_action,
        info=info
    )

@router.get("/state")
def get_hunt_state():
    if current_hunt_state is None:
        return {}
    return current_hunt_state.to_dict()

@router.post("/explain", response_model=HuntingExplainResponse)
def explain_decision(request: HuntingExplainRequest):
    # Explain the LAST decision or specific time step?
    # For simplicity, let's explain the current/last state decision.
    # We need to know what the state was.
    # If we track history, we can look up time step.
    
    if current_hunt_state is None:
        raise HTTPException(status_code=400, detail="No active hunt")
        
    # Get relevant state
    target_step = request.time_step if request.time_step is not None else current_hunt_state.time_step
    
    # Find in history
    step_data = None
    for item in current_hunt_state.history:
        if item["time_step"] == target_step:
            step_data = item
            break
            
    if not step_data:
        # If not found, maybe it's the current step before action?
        # Or just return empty
        return HuntingExplainResponse(explanation="Step not found in history", relevant_rules=[], q_values={})
        
    # Reconstruct Key
    # We need to know what the Impala Action was at that step.
    impala_act_val = step_data["impala_action"]
    lion_pos = step_data["lion_pos"]
    lion_st = step_data["lion_state"]
    
    # We need to map back to Enums
    # Assuming values match
    from app.core.entities import ImpalaAction, LionState
    impala_action = ImpalaAction(impala_act_val)
    lion_state = LionState(lion_st)
    
    state_key = training_manager.agent.get_state_key(lion_pos, impala_action, lion_state)
    
    # Get Q-Values
    q_values = training_manager.kb.q_table.get(state_key, {})
    
    # Find relevant abstractions
    rules = []
    for rule in training_manager.kb.abstractions:
        # Simple string matching or structured check?
        # Rule format: "IF Lion at {lion_pos} AND Lion is {lion_st} AND Impala does [{acts_str}] THEN {best_act}"
        # We check if our state matches the rule condition.
        # This parsing is brittle but fits the requirement.
        if f"Lion at {lion_pos[0]},{lion_pos[1]}" in rule and f"Lion is {lion_st}" in rule:
             if impala_act_val in rule:
                 rules.append(rule)
                 
    explanation = f"At step {target_step}, Lion was at {lion_pos} state {lion_st}. Impala did {impala_act_val}."
    if rules:
        explanation += " Decision influenced by general rules."
    else:
        explanation += " Decision based on specific Q-values."

    return HuntingExplainResponse(
        explanation=explanation,
        relevant_rules=rules,
        q_values=q_values
    )

@router.get("/result", response_model=HuntingResultResponse)
def get_hunt_result():
    if current_hunt_state is None:
        return HuntingResultResponse(result="unknown")
    return HuntingResultResponse(result=current_hunt_state.status)
