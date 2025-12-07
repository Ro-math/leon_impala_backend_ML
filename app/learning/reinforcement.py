import random
from typing import Tuple
from app.core.entities import LionAction, LionState, ImpalaAction
from app.learning.knowledge_base import KnowledgeBase
from app.learning.experience_replay import ExperienceReplay

class QLearningAgent:
    def __init__(self, knowledge_base: KnowledgeBase, 
                 learning_rate=0.3,  # Increased from 0.1 for faster learning
                 discount_factor=0.95,  # Increased from 0.9 for better long-term planning
                 epsilon_start=1.0,  # Start with full exploration
                 epsilon_end=0.05,  # Minimum exploration
                 epsilon_decay=0.995,  # Decay rate per episode
                 lambda_=0.8):  # Eligibility trace decay
        self.kb = knowledge_base
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.lambda_ = lambda_
        
        # Experience replay
        self.replay_buffer = ExperienceReplay(max_size=10000)
        
        # Eligibility traces
        self.eligibility_traces = {}

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
        for action in available_actions:
            q_val = self.kb.get_q_value(state_key, action.value)
            if q_val > max_q:
                max_q = q_val
                best_action = action
        
        # If all 0, pick random
        if max_q == 0.0 and best_action is None:
             return random.choice(available_actions)
             
        return best_action

    def learn(self, state_key: str, action: LionAction, reward: float, next_state_key: str, done: bool = False):
        """Standard Q-Learning update"""
        current_q = self.kb.get_q_value(state_key, action.value)
        
        # Max Q for next state
        if done:
            max_next_q = 0.0
        else:
            max_next_q = float('-inf')
            for a in LionAction:
                q = self.kb.get_q_value(next_state_key, a.value)
                if q > max_next_q:
                    max_next_q = q
            
            if max_next_q == float('-inf'): 
                max_next_q = 0.0
        
        # Q-Learning update rule
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        
        self.kb.update_q_value(state_key, action.value, new_q)
        
        # Add to replay buffer
        self.replay_buffer.add(state_key, action.value, reward, next_state_key, done)

    def learn_with_traces(self, state_key: str, action: LionAction, reward: float, next_state_key: str, done: bool = False):
        """Q-Learning with eligibility traces for faster credit assignment"""
        current_q = self.kb.get_q_value(state_key, action.value)
        
        # Calculate TD error
        if done:
            max_next_q = 0.0
        else:
            max_next_q = max([self.kb.get_q_value(next_state_key, a.value) for a in LionAction])
        
        td_error = reward + self.gamma * max_next_q - current_q
        
        # Update eligibility trace for current state-action
        trace_key = f"{state_key}|{action.value}"
        self.eligibility_traces[trace_key] = self.eligibility_traces.get(trace_key, 0.0) + 1.0
        
        # Update all state-actions with eligibility traces
        traces_to_remove = []
        for trace_key, eligibility in self.eligibility_traces.items():
            if eligibility > 0.01:  # Only update significant traces
                parts = trace_key.rsplit("|", 1)
                s_key = parts[0]
                a = parts[1]
                
                old_q = self.kb.get_q_value(s_key, a)
                new_q = old_q + self.alpha * td_error * eligibility
                self.kb.update_q_value(s_key, a, new_q)
                
                # Decay eligibility
                self.eligibility_traces[trace_key] *= self.gamma * self.lambda_
            else:
                traces_to_remove.append(trace_key)
        
        # Remove negligible traces
        for trace_key in traces_to_remove:
            del self.eligibility_traces[trace_key]
        
        # Add to replay buffer
        self.replay_buffer.add(state_key, action.value, reward, next_state_key, done)

    def learn_batch(self, batch_size=32):
        """Learn from a batch of experiences from replay buffer"""
        if self.replay_buffer.size() < batch_size:
            return
        
        batch = self.replay_buffer.sample(batch_size)
        for state_key, action, reward, next_state_key, done in batch:
            # Use standard Q-learning for batch updates (not traces)
            current_q = self.kb.get_q_value(state_key, action)
            
            if done:
                max_next_q = 0.0
            else:
                max_next_q = max([self.kb.get_q_value(next_state_key, a.value) for a in LionAction])
            
            new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
            self.kb.update_q_value(state_key, action, new_q)

    def reset_eligibility(self):
        """Reset eligibility traces at episode start"""
        self.eligibility_traces = {}

    def decay_epsilon(self):
        """Decay epsilon after each episode"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def get_epsilon(self) -> float:
        """Get current epsilon value"""
        return self.epsilon
