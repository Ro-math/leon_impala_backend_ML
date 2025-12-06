from enum import Enum
from typing import Tuple, Optional, ClassVar
from pydantic import BaseModel

class EntityType(str, Enum):
    LION = "lion"
    IMPALA = "impala"

class LionState(str, Enum):
    NORMAL = "normal"
    HIDDEN = "hidden"
    ATTACKING = "attacking"

class ImpalaState(str, Enum):
    NORMAL = "normal"
    DRINKING = "drinking"
    FLEEING = "fleeing"

class ImpalaAction(str, Enum):
    LOOK_LEFT = "look_left"
    LOOK_RIGHT = "look_right"
    LOOK_FRONT = "look_front"
    DRINK = "drink"
    FLEE = "flee"

class LionAction(str, Enum):
    ADVANCE = "advance"
    HIDE = "hide"
    ATTACK = "attack"
    # Positions 1-8 are setup actions, but during the game, these are the main ones.
    # We might handle initial placement separately.

class Entity(BaseModel):
    type: EntityType
    position: Tuple[int, int]

class Lion(Entity):
    type: EntityType = EntityType.LION
    state: LionState = LionState.NORMAL
    
    def move_towards(self, target: Tuple[int, int]):
        """Moves one step towards the target."""
        x, y = self.position
        tx, ty = target
        
        dx = tx - x
        dy = ty - y
        
        # Normalize to 1 step
        step_x = 0
        step_y = 0
        
        if abs(dx) >= abs(dy):
            if dx > 0: step_x = 1
            elif dx < 0: step_x = -1
            # If dy is also significant, we might move diagonally? 
            # "El león siempre avanza 1 cuadro en línea recta hacia el impala"
            # Usually implies Chebyshev distance or Euclidean rounded.
            # Let's assume 8-way movement is allowed if it's "straight line".
            # If strictly grid (Manhattan), it's different.
            # Given "19x19 map" and "triangles", diagonal seems likely.
            if dy > 0 and abs(dy) > abs(dx) * 0.5: step_y = 1
            elif dy < 0 and abs(dy) > abs(dx) * 0.5: step_y = -1
        else:
            if dy > 0: step_y = 1
            elif dy < 0: step_y = -1
            if dx > 0 and abs(dx) > abs(dy) * 0.5: step_x = 1
            elif dx < 0 and abs(dx) > abs(dy) * 0.5: step_x = -1

        # Simple approach: move to the neighbor that is closest to target
        # This is more robust
        best_dist = float('inf')
        best_pos = (x, y)
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                nx, ny = x + dx, y + dy
                dist = ((nx - tx)**2 + (ny - ty)**2)**0.5
                if dist < best_dist:
                    best_dist = dist
                    best_pos = (nx, ny)
        
        self.position = best_pos

class Impala(Entity):
    type: EntityType = EntityType.IMPALA
    state: ImpalaState = ImpalaState.NORMAL
    facing_direction: str = "north" # Fixed as per description "mirada siempre hacia el norte"
    # Actually, the prompt says "mirada siempre hacia el norte" BUT also "Ver a la izquierda", "Ver a la derecha".
    # This implies the "facing" is the base, and the "action" determines the current cone of vision.
    
    def flee(self, time_step_since_flee: int):
        """
        Updates position based on flee logic.
        Tn: 1 square
        Tn+1: 1 square
        Tn+2: 2 squares
        Tn+3: 3 squares
        ...
        Direction: East or West (straight line).
        """
        # Logic to decide East or West needs to be in GameEngine or here.
        # Usually away from danger.
        pass

class GameMap(BaseModel):
    width: int = 19
    height: int = 19
    waterhole_top_left: Tuple[int, int] = (6, 7)
    waterhole_bottom_right: Tuple[int, int] = (8, 11)
    
    # Initial Lion Positions
    # 1: (0,9), 2: (0,18), 3: (9,18), 4: (18,18), 5: (18,9), 6: (18,0), 7: (9,0), 8: (0,0) ??
    # Prompt says: 1:(0,9), 2:(0,18), 3:(9,18), 4:(18,18), 5:(18,9), 6:(18,0), 7:(9,0)
    # Maybe 8 is (0,0)? (0,0) is top left.
    # Let's assume 8 is (0,0) for now or check if I missed it.
    # Re-reading: "1 : (0,9) , 2 : (0,18) , 3 : (9,18) , 4 : (18,18) , 5 : (18,9) , 6 : (18 , 0), 7 : (9,0)"
    # It lists 7 points. But says "8 posiciones".
    # Let's look at the "figura 1" reference (which I don't have).
    # (0,9) is Left Middle.
    # (0,18) is Bottom Left.
    # (9,18) is Bottom Middle.
    # (18,18) is Bottom Right.
    # (18,9) is Right Middle.
    # (18,0) is Top Right.
    # (9,0) is Top Middle.
    # (0,0) Top Left is missing from the list but fits the pattern of perimeter points.
    # I will add (0,0) as point 8.
    
    valid_lion_positions: ClassVar[dict] = {
        1: (0, 9),
        2: (0, 18),
        3: (9, 18),
        4: (18, 18),
        5: (18, 9),
        6: (18, 0),
        7: (9, 0),
        8: (0, 0) 
    }
