from typing import Tuple
from app.utils.geometry import is_point_in_triangle
from app.core.entities import ImpalaAction, LionState

class VisionCalculator:
    def __init__(self):
        # Points definition
        # 1 : (0,9) , 2 : (0,18) , 3 : (9,18) , 4 : (18,18) , 5 : (18,9) , 6 : (18 , 0), 7 : (9,0)
        # 8 : (0,0) - Assumed based on context
        self.p2 = (0, 18)
        self.p4 = (18, 18)
        self.p6 = (18, 0)
        self.p8 = (0, 0)
        self.I = (9, 9)

    def is_lion_visible(self, lion_pos: Tuple[int, int], lion_state: LionState, impala_action: ImpalaAction) -> bool:
        """
        Determines if the lion is visible to the impala.
        """
        if lion_state == LionState.HIDDEN:
            return False
            
        if impala_action == ImpalaAction.LOOK_FRONT:
            # Triangle (8, I, 2) -> West
            return is_point_in_triangle(lion_pos, self.p8, self.I, self.p2)
            
        elif impala_action == ImpalaAction.LOOK_LEFT:
            # Triangle (8, I, 6) -> North
            return is_point_in_triangle(lion_pos, self.p8, self.I, self.p6)
            
        elif impala_action == ImpalaAction.LOOK_RIGHT:
            # Triangle (2, I, 4) -> South
            return is_point_in_triangle(lion_pos, self.p2, self.I, self.p4)
            
        elif impala_action == ImpalaAction.DRINK:
            # Can only see reflection? Prompt says: "cuando está bebiendo sólo puede ver su reflejo en el agua"
            # Implies it cannot see the lion.
            return False
            
        elif impala_action == ImpalaAction.FLEE:
            # When fleeing, does it see?
            # "Una vez que el impala comienza a huir no podrá realizar otra acción."
            # Usually fleeing implies the game is ending or it's running away.
            # If it's already fleeing, visibility might not matter for triggering flee (already triggered).
            return True # Assume it sees everything or nothing, but logic elsewhere handles flee trigger.

        return False
