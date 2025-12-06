from typing import Dict, List
from app.core.entities import ImpalaAction

class AbstractionEngine:
    def __init__(self, knowledge_base):
        self.kb = knowledge_base

    def abstract_knowledge(self):
        """
        Scans the Q-Table and finds patterns to generalize.
        Example: If (Pos1, LookLeft, Normal) -> Advance is good
                 AND (Pos1, LookRight, Normal) -> Advance is good
                 THEN (Pos1, LookSide, Normal) -> Advance is good.
        """
        # This is a simplified abstraction logic.
        # We look for states that differ only by ImpalaAction and have the same best action.
        
        # Group by (LionPos, LionState)
        groups = {}
        
        for state_key, actions in self.kb.q_table.items():
            # Parse key: "LionPos_ImpalaAction_LionState"
            # We need a consistent key format. 
            # Let's assume key is "x,y|impala_action|lion_state"
            parts = state_key.split("|")
            if len(parts) != 3: continue
            
            lion_pos = parts[0]
            impala_act = parts[1]
            lion_st = parts[2]
            
            # Find best action
            best_action = max(actions, key=actions.get)
            best_val = actions[best_action]
            
            if best_val > 0: # Only abstract positive knowledge
                context_key = f"{lion_pos}|{lion_st}"
                if context_key not in groups:
                    groups[context_key] = []
                groups[context_key].append((impala_act, best_action))

        # Analyze groups
        new_abstractions = []
        for context, items in groups.items():
            # items is list of (impala_action, best_action)
            # Check if we have multiple impala actions leading to same best action
            action_map = {}
            for imp_act, best_act in items:
                if best_act not in action_map:
                    action_map[best_act] = []
                action_map[best_act].append(imp_act)
            
            for best_act, imp_acts in action_map.items():
                if len(imp_acts) > 1:
                    # We found a generalization!
                    lion_pos, lion_st = context.split("|")
                    acts_str = ", ".join(imp_acts)
                    rule = f"IF Lion at {lion_pos} AND Lion is {lion_st} AND Impala does [{acts_str}] THEN {best_act}"
                    if rule not in self.kb.abstractions:
                        new_abstractions.append(rule)
        
        self.kb.abstractions.extend(new_abstractions)
        return new_abstractions
