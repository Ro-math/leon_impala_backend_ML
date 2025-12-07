"""
Unit tests for Q-learning agent and training
Tests agent behavior, learning updates, and training episodes
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from app.learning.knowledge_base import KnowledgeBase
from app.learning.reinforcement import QLearningAgent
from app.core.entities import LionAction, ImpalaAction, LionState, GameMap
from app.core.game_engine import GameEngine, GameState


class TestQLearningAgent:
    """Tests for Q-learning agent behavior"""
    
    def test_agent_init(self):
        """Test agent initializes with correct parameters"""
        kb = KnowledgeBase()
        agent = QLearningAgent(kb, learning_rate=0.2, discount_factor=0.95, exploration_rate=0.15)
        
        assert agent.alpha == 0.2
        assert agent.gamma == 0.95
        assert agent.epsilon == 0.15
        assert agent.kb == kb
    
    def test_get_state_key(self):
        """Test state key generation"""
        kb = KnowledgeBase()
        agent = QLearningAgent(kb)
        
        state_key = agent.get_state_key((0, 9), ImpalaAction.DRINK, LionState.NORMAL)
        
        assert state_key == "0,9|drink|normal"
        assert isinstance(state_key, str)
    
    def test_agent_choose_action_exploitation(self):
        """Test agent exploits best Q-value when not exploring"""
        kb = KnowledgeBase()
        agent = QLearningAgent(kb, exploration_rate=0.0)  # No exploration
        
        state_key = "test_state"
        kb.update_q_value(state_key, "advance", 10.0)
        kb.update_q_value(state_key, "hide", 5.0)
        kb.update_q_value(state_key, "attack", 2.0)
        
        # Should always choose advance (highest Q-value)
        for _ in range(10):
            action = agent.choose_action(state_key)
            assert action == LionAction.ADVANCE
    
    def test_agent_choose_action_exploration(self):
        """Test agent explores with epsilon probability"""
        kb = KnowledgeBase()
        agent = QLearningAgent(kb, exploration_rate=1.0)  # Always explore
        
        state_key = "test_state"
        kb.update_q_value(state_key, "advance", 100.0)  # Clear best action
        
        # Should sometimes choose non-optimal actions
        actions = set()
        for _ in range(20):
            action = agent.choose_action(state_key)
            actions.add(action)
        
        # Should have explored different actions
        assert len(actions) >= 2  # Not always the same action


class TestQLearningUpdate:
    """Tests for Q-learning update formula"""
    
    def test_agent_learn_updates_q_value(self):
        """Test Q-learning update formula"""
        kb = KnowledgeBase()
        agent = QLearningAgent(kb, learning_rate=0.1, discount_factor=0.9)
        
        state_key = "current_state"
        next_state_key = "next_state"
        action = LionAction.ADVANCE
        reward = 10.0
        
        # Set next state Q-values
        kb.update_q_value(next_state_key, "advance", 5.0)
        kb.update_q_value(next_state_key, "hide", 3.0)
        
        # Initial Q-value
        initial_q = kb.get_q_value(state_key, action.value)
        assert initial_q == 0.0
        
        # Learn
        agent.learn(state_key, action, reward, next_state_key)
        
        # Q-value should have updated
        new_q = kb.get_q_value(state_key, action.value)
        
        # Expected: 0 + 0.1 * (10 + 0.9 * 5 - 0) = 0 + 0.1 * 14.5 = 1.45
        assert new_q == pytest.approx(1.45, rel=0.01)
    
    def test_agent_learn_terminal_state(self):
        """Test learning with terminal state"""
        kb = KnowledgeBase()
        agent = QLearningAgent(kb, learning_rate=0.5, discount_factor=0.9)
        
        state_key = "current_state"
        action = LionAction.ATTACK
        reward = 100.0  # Success reward
        
        # Learn with terminal state
        agent.learn(state_key, action, reward, "TERMINAL")
        
        # Q-value should be updated considering no future reward
        # Expected: 0 + 0.5 * (100 + 0 - 0) = 50.0
        new_q = kb.get_q_value(state_key, action.value)
        assert new_q == pytest.approx(50.0, rel=0.01)
    
    def test_agent_learn_multiple_updates(self):
        """Test Q-value converges with multiple updates"""
        kb = KnowledgeBase()
        agent = QLearningAgent(kb, learning_rate=0.1, discount_factor=0.9)
        
        state_key = "state"
        next_state_key = "next"
        action = LionAction.ADVANCE
        
        kb.update_q_value(next_state_key, "advance", 10.0)
        
        # Multiple learning steps with same reward
        for _ in range(100):
            agent.learn(state_key, action, 5.0, next_state_key)
        
        # Should converge toward reward + gamma * max_next_q
        # = 5 + 0.9 * 10 = 14
        final_q = kb.get_q_value(state_key, action.value)
        assert final_q == pytest.approx(14.0, rel=0.1)


class TestTrainingEpisode:
    """Tests for complete training episodes"""
    
    def test_training_episode_completion(self):
        """Test a complete training episode runs"""
        kb = KnowledgeBase()
        agent = QLearningAgent(kb, exploration_rate=0.1)
        engine = GameEngine()
        
        # Initialize state
        start_pos = GameMap.valid_lion_positions[1]
        state = GameState(lion_start_pos=start_pos)
        
        done = False
        steps = 0
        max_steps = 50
        
        while not done and steps < max_steps:
            # Get current state key
            state_key = agent.get_state_key(
                state.lion.position,
                ImpalaAction.DRINK,  # Simplified
                state.lion.state
            )
            
            # Choose action
            lion_action = agent.choose_action(state_key)
            
            # Execute
            next_state, reward, done, info = engine.step(
                state, lion_action, ImpalaAction.DRINK
            )
            
            # Learn
            next_state_key = "TERMINAL" if done else "NEXT_KEY_TODO"
            agent.learn(state_key, lion_action, reward, next_state_key)
            
            state = next_state
            steps += 1
        
        # Should have some Q-values learned
        assert len(kb.q_table) > 0
        assert steps < max_steps or done
    
    def test_q_values_improve_over_time(self):
        """Test Q-values improve with training"""
        kb = KnowledgeBase()
        agent = QLearningAgent(kb, learning_rate=0.1, exploration_rate=0.2)
        engine = GameEngine()
        
        success_count = 0
        episodes = 10
        
        for episode in range(episodes):
            state = GameState(lion_start_pos=(9, 7))  # Close to impala
            done = False
            steps = 0
            
            while not done and steps < 20:
                state_key = agent.get_state_key(
                    state.lion.position,
                    ImpalaAction.DRINK,
                    state.lion.state
                )
                
                lion_action = agent.choose_action(state_key)
                next_state, reward, done, info = engine.step(
                    state, lion_action, ImpalaAction.DRINK
                )
                
                agent.learn(state_key, lion_action, reward, "TERMINAL" if done else "NEXT")
                
                state = next_state
                steps += 1
            
            if state.status == "success":
                success_count += 1
        
        # With a close starting position, agent should learn to succeed
        # At least some episodes should be successful
        assert success_count > 0
        assert len(kb.q_table) > 0


class TestAgentDecisionMaking:
    """Tests for agent decision making"""
    
    def test_agent_prefers_higher_q_values(self):
        """Test agent chooses actions with higher Q-values"""
        kb = KnowledgeBase()
        agent = QLearningAgent(kb, exploration_rate=0.0)
        
        state_key = "decision_state"
        kb.update_q_value(state_key, "advance", 5.0)
        kb.update_q_value(state_key, "hide", 15.0)  # Best
        kb.update_q_value(state_key, "attack", 3.0)
        
        action = agent.choose_action(state_key)
        assert action == LionAction.HIDE
    
    def test_agent_handles_negative_q_values(self):
        """Test agent chooses least negative option"""
        kb = KnowledgeBase()
        agent = QLearningAgent(kb, exploration_rate=0.0)
        
        state_key = "bad_state"
        kb.update_q_value(state_key, "advance", -10.0)
        kb.update_q_value(state_key, "hide", -5.0)  # Least bad
        kb.update_q_value(state_key, "attack", -20.0)
        
        action = agent.choose_action(state_key)
        assert action == LionAction.HIDE
    
    def test_agent_handles_all_zero_q_values(self):
        """Test agent chooses randomly when all Q-values are equal"""
        kb = KnowledgeBase()
        agent = QLearningAgent(kb, exploration_rate=0.0)
        
        state_key = "neutral_state"
        # All default to 0.0
        
        # Should still choose an action (random tie-breaking)
        action = agent.choose_action(state_key)
        assert action in [LionAction.ADVANCE, LionAction.HIDE, LionAction.ATTACK]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
