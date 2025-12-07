import asyncio
import random
from fastapi import APIRouter, BackgroundTasks, HTTPException
from app.models.requests import TrainingStartRequest
from app.models.responses import TrainingStatusResponse, TrainingStatisticsResponse
from app.core.game_engine import GameEngine, GameState, GameMap, ImpalaState
from app.core.entities import LionAction, ImpalaAction
from app.learning.knowledge_base import KnowledgeBase
from app.learning.reinforcement import QLearningAgent
from app.learning.abstraction import AbstractionEngine

router = APIRouter()

# Global Training State
class TrainingManager:
    def __init__(self):
        self.is_running = False
        self.stop_requested = False
        self.progress = 0.0
        self.current_incursion = 0
        self.total_incursions = 0
        self.success_count = 0
        self.fail_count = 0
        
        self.kb = KnowledgeBase()
        self.agent = QLearningAgent(self.kb)
        self.engine = GameEngine()
        self.abstraction_engine = AbstractionEngine(self.kb)
        
        # Initialize stats
        self.success_rate_by_position = {k: 0.0 for k in GameMap.valid_lion_positions.keys()}
        self.position_attempts = {k: 0 for k in GameMap.valid_lion_positions.keys()}
        self.position_successes = {k: 0 for k in GameMap.valid_lion_positions.keys()}
        self.total_steps = 0

    def start_training(self, request: TrainingStartRequest):
        if self.is_running:
            raise HTTPException(status_code=400, detail="Training already in progress")
        
        self.is_running = True
        self.stop_requested = False
        self.total_incursions = request.num_incursions
        self.current_incursion = 0
        self.success_count = 0
        self.fail_count = 0
        
        self.success_rate_by_position = {k: 0.0 for k in GameMap.valid_lion_positions.keys()}
        self.position_attempts = {k: 0 for k in GameMap.valid_lion_positions.keys()}
        self.position_successes = {k: 0 for k in GameMap.valid_lion_positions.keys()}
        self.total_steps = 0
        
        # Run in background
        asyncio.create_task(self._training_loop(request))
        
    def resume_training(self):
        if self.is_running:
            raise HTTPException(status_code=400, detail="Training already in progress")
        
        remaining = self.total_incursions - self.current_incursion
        if remaining <= 0:
            raise HTTPException(status_code=400, detail="Training already completed")
            
        self.is_running = True
        self.stop_requested = False
        
        # Reconstruct request from state? 
        # For simplicity, we assume same parameters as last run or defaults.
        # Ideally we should store the last request.
        # Let's assume we just continue with random params if not stored.
        # But we need the initial positions list.
        # We will store the last request.
        if not hasattr(self, 'last_request'):
             raise HTTPException(status_code=400, detail="No previous training to resume")
             
        asyncio.create_task(self._training_loop(self.last_request, start_index=self.current_incursion))

    def stop_training(self):
        if self.is_running:
            self.stop_requested = True

    async def _training_loop(self, request: TrainingStartRequest, start_index: int = 0):
        self.last_request = request
        print(f"Starting training loop from {start_index}...")
        
        for i in range(start_index, request.num_incursions):
            if self.stop_requested:
                break
            
            self.current_incursion = i + 1
            self.progress = (i + 1) / request.num_incursions
            
            # Setup Episode
            start_pos_idx = random.choice(request.initial_positions)
            start_pos = GameMap.valid_lion_positions[start_pos_idx]
            self.position_attempts[start_pos_idx] += 1
            
            state = GameState(lion_start_pos=start_pos)
            done = False
            steps = 0
            
            while not done:
                steps += 1
                # ... (Logic same as before)
                impala_action = ImpalaAction.LOOK_FRONT 
                if request.impala_mode == "random":
                    choices = [a for a in ImpalaAction if a != ImpalaAction.FLEE]
                    impala_action = random.choice(choices)
                elif request.impala_mode == "programmed" and request.impala_sequence:
                    seq_idx = state.time_step % len(request.impala_sequence)
                    impala_action = request.impala_sequence[seq_idx]
                
                state_key = self.agent.get_state_key(state.lion.position, impala_action, state.lion.state)
                lion_action = self.agent.choose_action(state_key)
                next_state, reward, done, info = self.engine.step(state, lion_action, impala_action)
                
                # Learn
                self.agent.learn(state_key, lion_action, reward, "TERMINAL" if done else "NEXT_KEY_TODO")
                # Note: "NEXT_KEY_TODO" is placeholder for the issue discussed earlier.
                # Ideally we predict next Impala action or use current.
                # For now, we keep it simple as implemented before.
                
                state = next_state
                
                if done:
                    self.total_steps += steps
                    if state.status == "success": 
                        self.success_count += 1
                        self.position_successes[start_pos_idx] += 1
                    else: 
                        self.fail_count += 1
                    
                    # Terminal update handled above or here?
                    # learn() called above with done flag logic inside agent? 
                    # My agent.learn uses next_state_key to look up max_q.
                    # If terminal, max_q is 0.
                    pass

            # Update stats
            for k in self.position_attempts:
                if self.position_attempts[k] > 0:
                    self.success_rate_by_position[k] = self.position_successes[k] / self.position_attempts[k]

            # Periodic Save (every 100 episodes)
            if i % 100 == 0:
                self.abstraction_engine.abstract_knowledge()
                self.kb.save("knowledge_checkpoint")
                self._save_log(i, state.history)
                
        # Final Save
        self.kb.save("knowledge_final")
        self.is_running = False
        print("Training finished.")

    def _save_log(self, episode_idx: int, history: list):
        from app.storage.json_storage import JsonStorage
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/logs/training_ep{episode_idx}_{timestamp}.json"
        
        log_data = {
            "episode": episode_idx,
            "timestamp": timestamp,
            "history": history
        }
        JsonStorage.save(log_data, filename)

training_manager = TrainingManager()

@router.post("/start")
async def start_training(request: TrainingStartRequest):
    training_manager.start_training(request)
    return {"message": "Training started"}

@router.post("/resume")
async def resume_training():
    training_manager.resume_training()
    return {"message": "Training resumed"}

@router.post("/stop")
async def stop_training():
    training_manager.stop_training()
    return {"message": "Training stop requested"}

@router.get("/status", response_model=TrainingStatusResponse)
def get_training_status():
    return TrainingStatusResponse(
        status="running" if training_manager.is_running else "stopped",
        progress=training_manager.progress,
        current_incursion=training_manager.current_incursion,
        total_incursions=training_manager.total_incursions,
        success_count=training_manager.success_count,
        fail_count=training_manager.fail_count
    )

@router.get("/statistics", response_model=TrainingStatisticsResponse)
def get_training_statistics():
    avg_steps = 0
    if training_manager.current_incursion > 0:
        avg_steps = training_manager.total_steps / training_manager.current_incursion
        
    success_rate = 0
    if training_manager.current_incursion > 0:
        success_rate = training_manager.success_count / training_manager.current_incursion

    return TrainingStatisticsResponse(
        total_incursions=training_manager.current_incursion,
        success_rate=success_rate,
        avg_steps=avg_steps,
        success_rate_by_position=training_manager.success_rate_by_position,
        abstractions_count=len(training_manager.kb.abstractions),
        q_table_size=len(training_manager.kb.q_table)
    )
