from typing import Tuple, List, Optional
from app.core.entities import Lion, Impala, GameMap, LionAction, ImpalaAction, LionState, ImpalaState
from app.core.vision_calculator import VisionCalculator
from app.utils.geometry import calculate_distance

class GameState:
    def __init__(self, lion_start_pos: Tuple[int, int]):
        self.map = GameMap()
        self.lion = Lion(position=lion_start_pos)
        self.impala = Impala(position=(9, 9))
        self.time_step = 0
        self.history = []
        self.status = "in_progress" # in_progress, success, failed
        self.flee_start_time = -1

    def to_dict(self):
        return {
            "lion": self.lion.dict(),
            "impala": self.impala.dict(),
            "time_step": self.time_step,
            "status": self.status
        }

class GameEngine:
    def __init__(self):
        self.vision_calculator = VisionCalculator()

    def step(self, state: GameState, lion_action: LionAction, impala_action: ImpalaAction) -> Tuple[GameState, float, bool, str]:
        """
        Executes one time step.
        Returns: New State, Reward, Done, Info
        """
        # 1. Impala acts first (conceptually parallel, but we need Impala's action to check visibility)
        # Actually prompt says: "Siempre, primero el impala actúa y el león reacciona... Una vez realizadas... el sistema verifica"
        
        state.time_step += 1
        info = ""
        
        # Update Impala State/Action
        # If already fleeing, override action
        if state.impala.state == ImpalaState.FLEEING:
            impala_action = ImpalaAction.FLEE
            self._handle_flee_movement(state)
        else:
            if impala_action == ImpalaAction.DRINK:
                state.impala.state = ImpalaState.DRINKING
            else:
                state.impala.state = ImpalaState.NORMAL
        
        # 2. Lion acts
        # If Lion attacks, it moves 2 squares
        # If Lion advances, 1 square
        # If Lion hides, state = HIDDEN
        
        # Enforce Attack Persistence
        # "Una vez que el león inicia un ataque no podrá realizar otra acción."
        if state.lion.state == LionState.ATTACKING:
            lion_action = LionAction.ATTACK
            
        prev_lion_pos = state.lion.position
        
        if lion_action == LionAction.ATTACK:
            state.lion.state = LionState.ATTACKING
            # Move 2 squares towards Impala
            state.lion.move_towards(state.impala.position)
            state.lion.move_towards(state.impala.position) # Twice for speed 2
        elif lion_action == LionAction.HIDE:
            state.lion.state = LionState.HIDDEN
        elif lion_action == LionAction.ADVANCE:
            state.lion.state = LionState.NORMAL
            state.lion.move_towards(state.impala.position)
        
        # 3. Verify World Modifications & Flee Conditions
        
        # Check Distance
        dist = calculate_distance(state.lion.position, state.impala.position)
        
        # Condition 1: Impala sees Lion
        is_visible = self.vision_calculator.is_lion_visible(state.lion.position, state.lion.state, impala_action)
        
        # Condition 2: Lion attacks (Impala detects attack immediately? "Cuando el león comienza un ataque")
        is_attacking = (lion_action == LionAction.ATTACK)
        
        # Condition 3: Distance < 3
        is_too_close = (dist < 3)
        
        flee_triggered = False
        reason = ""
        
        if state.impala.state != ImpalaState.FLEEING:
            if is_visible:
                flee_triggered = True
                reason = "Lion visible"
            elif is_attacking:
                flee_triggered = True
                reason = "Lion attacking"
            elif is_too_close:
                flee_triggered = True
                reason = "Too close"
                
            if flee_triggered:
                state.impala.state = ImpalaState.FLEEING
                state.flee_start_time = state.time_step
                info = f"Flee triggered: {reason}"

        # Check End Conditions
        done = False
        reward = 0.0
        
        # Success: Lion reaches Impala (Distance < 1 or same square?)
        # "El león alcanza al impala"
        # Success: Lion reaches Impala (Distance < 1 or same square?)
        # "El león alcanza al impala"
        if dist <= 1.0: # Same square or adjacent
            state.status = "success"
            done = True
            reward = 100.0
            info += " Lion caught Impala!"
            
        # Failure: Impala flees and Lion cannot catch
        # "El león no podrá alcanzar al impala"
        # If Impala is fleeing, we need to check if Lion can catch.
        # Impala speed increases: 1, 1, 2, 3...
        # Lion attack speed: 2.
        # If Impala is fleeing, it's likely game over unless Lion is VERY close.
        if state.impala.state == ImpalaState.FLEEING:
            # If Lion didn't catch it this turn, check if it's possible.
            # For simplicity, if Impala is fleeing and not caught yet, we might count as fail soon.
            # But let's let it run a few steps?
            # Prompt: "La incursión de cacería termina cuando el sistema detecta que: El león no podrá alcanzar al impala."
            # If Impala speed > Lion speed and distance is increasing, fail.
            
            flee_duration = state.time_step - state.flee_start_time
            impala_speed = self._get_impala_speed(flee_duration)
            
            if impala_speed > 2 and dist > 1:
                state.status = "failed"
                done = True
                reward = -100.0
                info += " Impala escaped."
            elif dist > 10: # Arbitrary cutoff
                state.status = "failed"
                done = True
                reward = -100.0
        
        # Step penalty
        if not done:
            reward -= 1.0
            
        # Update History
        state.history.append({
            "time_step": state.time_step,
            "lion_pos": state.lion.position,
            "impala_pos": state.impala.position,
            "lion_state": state.lion.state,
            "impala_state": state.impala.state,
            "lion_action": lion_action.value,
            "impala_action": impala_action.value,
            "info": info
        })
            
        return state, reward, done, info

    def _handle_flee_movement(self, state: GameState):
        duration = state.time_step - state.flee_start_time
        speed = self._get_impala_speed(duration)
        
        # Move East or West.
        # User confirmed (row, col) system. East/West is Column (index 1).
        # Move away from Lion's column.
        dy = state.lion.position[1] - state.impala.position[1]
        
        # If Lion is same col, pick random or default? 
        # Usually flee away. If dy > 0 (Lion is "East"/Right), go West (-1).
        # If dy < 0 (Lion is "West"/Left), go East (1).
        # If dy == 0, pick one? Let's say East.
        direction = -1 if dy > 0 else 1 
        
        # Update Y (Column)
        new_y = state.impala.position[1] + (direction * speed)
        
        # Clamp to map 19x19 (0-18)
        new_y = max(0, min(18, new_y))
        
        state.impala.position = (state.impala.position[0], new_y)
        
        # Update Facing Direction
        state.impala.facing_direction = "east" if direction == 1 else "west"

    def _get_impala_speed(self, duration: int) -> int:
        # Impala acceleration when fleeing:
        # Tn (duration=0): speed = 1
        # Tn+1 (duration=1): speed = 2
        # Tn+2 (duration=2): speed = 3
        # Tn+3 (duration=3): speed = 4
        # etc.
        return duration + 1
