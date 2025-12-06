import random
from typing import Tuple
from app.core.entities import LionAction, LionState, ImpalaAction
from app.learning.knowledge_base import KnowledgeBase

class QLearningAgent:
    def __init__(self, knowledge_base: KnowledgeBase, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.kb = knowledge_base
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate

    def get_state_key(self, lion_pos: Tuple[int, int], impala_action: ImpalaAction, lion_state: LionState) -> str:
        # Key format: "x,y|impala_action|lion_state"
        return f"{lion_pos[0]},{lion_pos[1]}|{impala_action.value}|{lion_state.value}"

    def choose_action(self, state_key: str, available_actions: list = None) -> LionAction:
        if available_actions is None:
            available_actions = list(LionAction)

        if random.random() < self.epsilon:
            return random.choice(available_actions)
        
        # Exploitation: choose best Q-value
        best_action = None
        max_q = float('-inf')
        
        # Check Q-values for all actions
        # If state not in KB, it returns 0.0
        # We need to iterate over available actions to find max
        for action in available_actions:
            q_val = self.kb.get_q_value(state_key, action.value)
            if q_val > max_q:
                max_q = q_val
                best_action = action
        
        # If all 0, pick random
        if max_q == 0.0 and best_action is None:
             return random.choice(available_actions)
             
        return best_action

    def learn(self, state_key: str, action: LionAction, reward: float, next_state_key: str):
        current_q = self.kb.get_q_value(state_key, action.value)
        
        # Max Q for next state
        max_next_q = float('-inf')
        for a in LionAction:
            q = self.kb.get_q_value(next_state_key, a.value)
            if q > max_next_q:
                max_next_q = q
        
        if max_next_q == float('-inf'): max_next_q = 0.0
        
        # Q-Learning update rule
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        
        self.kb.update_q_value(state_key, action.value, new_q)
