from app.core.entities import LionState, ImpalaState

class RewardSystem:
    def calculate_reward(self, prev_state, action, new_state, done, info):
        # This logic is already partly in GameEngine, but we can refine it here.
        # Or GameEngine returns a raw reward and this system adjusts it?
        # For Q-Learning, we usually take the reward from the environment (GameEngine).
        # So this class might be redundant if GameEngine handles it.
        # However, to keep architecture clean:
        
        reward = 0.0
        
        if new_state.status == "success":
            reward = 100.0
        elif new_state.status == "failed":
            reward = -100.0
        else:
            # Step penalty to encourage speed
            reward = -1.0
            
            # Additional shaping?
            # If Lion gets closer?
            # dist_prev = ...
            # dist_new = ...
            # if dist_new < dist_prev: reward += 0.1
            pass
            
        return reward
