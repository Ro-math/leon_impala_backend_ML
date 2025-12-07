from app.core.entities import LionAction, LionState, ImpalaState
from app.utils.geometry import calculate_distance

class RewardSystem:
    def calculate_reward(self, prev_state, action, new_state, done, info):
        """
        Calculate reward with shaping for faster learning.
        Includes distance-based incentives and strategic action rewards.
        """
        reward = 0.0
        
        # Terminal rewards
        if new_state.status == "success":
            return 100.0
        elif new_state.status == "failed":
            return -100.0
        
        # Distance-based reward shaping
        prev_dist = calculate_distance(prev_state.lion.position, prev_state.impala.position)
        new_dist = calculate_distance(new_state.lion.position, new_state.impala.position)
        
        # Reward for getting closer (positive reinforcement)
        if new_dist < prev_dist:
            reward += 2.0
        elif new_dist > prev_dist:
            reward -= 2.0  # Penalty for moving away
        
        # Strategic action rewards
        if action == LionAction.HIDE and new_dist > 5:
            reward += 1.0  # Good to hide when far
        elif action == LionAction.ATTACK and new_dist <= 3:
            reward += 3.0  # Good to attack when close
        elif action == LionAction.ATTACK and new_dist > 5:
            reward -= 2.0  # Bad to attack when too far
        
        # Encourage advancing when at medium distance
        if action == LionAction.ADVANCE and 3 < new_dist <= 6:
            reward += 1.0
        
        # Penalty for triggering flee too early
        if new_state.impala.state == ImpalaState.FLEEING and prev_state.impala.state != ImpalaState.FLEEING:
            if new_dist > 5:
                reward -= 5.0  # Big penalty for scaring impala when far
        
        # Step penalty (encourage efficiency)
        reward -= 0.5
        
        return reward
