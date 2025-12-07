import random
from collections import deque
from typing import Tuple, List

class ExperienceReplay:
    """Experience replay buffer for storing and sampling past experiences"""
    
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state_key: str, action: str, reward: float, next_state_key: str, done: bool):
        """Add experience to buffer"""
        self.buffer.append((state_key, action, reward, next_state_key, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample random batch from buffer"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(self.buffer, batch_size)
    
    def size(self) -> int:
        """Return current buffer size"""
        return len(self.buffer)
    
    def clear(self):
        """Clear all experiences"""
        self.buffer.clear()
