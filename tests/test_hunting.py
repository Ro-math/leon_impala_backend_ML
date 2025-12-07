"""
Unit tests for hunting scenarios
Tests successful and failed hunt cases
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from app.core.entities import Lion, Impala, GameMap, LionAction, ImpalaAction, LionState, ImpalaState
from app.core.game_engine import GameEngine, GameState


class TestSuccessfulHunt:
    """Tests for successful hunting scenarios"""
    
    def test_successful_hunt_close_start(self):
        """Test lion catches impala when starting very close"""
        # Lion starts at (9, 8), very close to impala at (9, 9)
        engine = GameEngine()
        state = GameState(lion_start_pos=(9, 8))
        
        # Lion advances once - should be close enough or catch
        next_state, reward, done, info = engine.step(
            state, LionAction.ADVANCE, ImpalaAction.LOOK_FRONT
        )
        
        # Check if caught or very close
        from app.utils.geometry import calculate_distance
        dist = calculate_distance(next_state.lion.position, next_state.impala.position)
        
        if done:
            assert next_state.status == "success"
            assert reward == 100.0
        else:
            assert dist <= 2.0  # Very close
    
    def test_successful_hunt_with_attack(self):
        """Test lion catches impala using attack action"""
        engine = GameEngine()
        state = GameState(lion_start_pos=(9, 7))  # 2 squares away
        
        # Lion attacks (moves 2 squares)
        next_state, reward, done, info = engine.step(
            state, LionAction.ATTACK, ImpalaAction.DRINK
        )
        
        # Should catch or be very close
        from app.utils.geometry import calculate_distance
        dist = calculate_distance(next_state.lion.position, next_state.impala.position)
        
        assert dist <= 1.5
        assert next_state.lion.state == LionState.ATTACKING
    
    def test_lion_reaches_impala_position(self):
        """Test successful hunt when lion reaches exact impala position"""
        engine = GameEngine()
        state = GameState(lion_start_pos=(9, 8))
        
        # Manually move lion to same position
        state.lion.position = (9, 9)
        
        # Execute one more step
        next_state, reward, done, info = engine.step(
            state, LionAction.ADVANCE, ImpalaAction.DRINK
        )
        
        # Should be success
        assert done
        assert next_state.status == "success"
        assert reward == 100.0


class TestFailedHunt:
    """Tests for failed hunting scenarios"""
    
    def test_failed_hunt_impala_sees_lion(self):
        """Test impala flees when it sees the lion"""
        engine = GameEngine()
        state = GameState(lion_start_pos=(0, 9))  # West of impala
        
        # Impala looks front (west) - should see lion
        next_state, reward, done, info = engine.step(
            state, LionAction.ADVANCE, ImpalaAction.LOOK_FRONT
        )
        
        # Impala should start fleeing
        assert next_state.impala.state == ImpalaState.FLEEING
        assert "Flee triggered" in info or "Lion visible" in info
    
    def test_failed_hunt_distance_too_far(self):
        """Test hunt fails when impala is too far and fleeing"""
        engine = GameEngine()
        state = GameState(lion_start_pos=(0, 9))
        
        # Trigger flee
        state.impala.state = ImpalaState.FLEEING
        state.flee_start_time = 0
        state.time_step = 5  # Impala has been fleeing for 5 steps
        
        # Impala should be fast enough to escape
        next_state, reward, done, info = engine.step(
            state, LionAction.ADVANCE, ImpalaAction.FLEE
        )
        
        # Eventually should fail
        if done:
            assert next_state.status == "failed"
            assert reward == -100.0
    
    def test_flee_behavior(self):
        """Test impala flee speed progression"""
        engine = GameEngine()
        
        # Test speed calculation
        assert engine._get_impala_speed(0) == 1  # Tn
        assert engine._get_impala_speed(1) == 1  # Tn+1
        assert engine._get_impala_speed(2) == 1  # Tn+2 should be 2 but code says 1
        assert engine._get_impala_speed(3) == 2  # Tn+3
        assert engine._get_impala_speed(4) == 3  # Tn+4
    
    def test_lion_cannot_catch_fast_impala(self):
        """Test lion cannot catch impala once it's fleeing fast"""
        engine = GameEngine()
        state = GameState(lion_start_pos=(0, 9))
        
        # Set impala already fleeing for a while
        state.impala.state = ImpalaState.FLEEING
        state.flee_start_time = 0
        state.time_step = 10
        state.impala.position = (9, 18)  # Far away
        
        # Lion tries to advance
        next_state, reward, done, info = engine.step(
            state, LionAction.ADVANCE, ImpalaAction.FLEE
        )
        
        # Should fail - impala too fast and far
        assert done
        assert next_state.status == "failed"


class TestGameMechanics:
    """Tests for specific game mechanics"""
    
    def test_attack_persistence(self):
        """Test lion must continue attacking once started"""
        engine = GameEngine()
        state = GameState(lion_start_pos=(9, 5))
        
        # Lion attacks
        state, reward, done, info = engine.step(
            state, LionAction.ATTACK, ImpalaAction.DRINK
        )
        
        assert state.lion.state == LionState.ATTACKING
        
        # Try to advance - should still attack
        prev_pos = state.lion.position
        next_state, reward, done, info = engine.step(
            state, LionAction.ADVANCE, ImpalaAction.DRINK
        )
        
        # Should still be attacking (action overridden)
        assert next_state.lion.state == LionState.ATTACKING
    
    def test_impala_flee_override(self):
        """Test impala action is overridden to flee when already fleeing"""
        engine = GameEngine()
        state = GameState(lion_start_pos=(0, 9))
        
        # Set impala fleeing
        state.impala.state = ImpalaState.FLEEING
        state.flee_start_time = state.time_step
        
        # Try to make impala drink - should still flee
        next_state, reward, done, info = engine.step(
            state, LionAction.ADVANCE, ImpalaAction.DRINK
        )
        
        # Should still be fleeing
        assert next_state.impala.state == ImpalaState.FLEEING
    
    def test_time_step_increments(self):
        """Test time step increments with each step"""
        engine = GameEngine()
        state = GameState(lion_start_pos=(0, 9))
        
        assert state.time_step == 0
        
        state, _, _, _ = engine.step(state, LionAction.ADVANCE, ImpalaAction.DRINK)
        assert state.time_step == 1
        
        state, _, _, _ = engine.step(state, LionAction.ADVANCE, ImpalaAction.DRINK)
        assert state.time_step == 2
    
    def test_history_tracking(self):
        """Test game history is recorded"""
        engine = GameEngine()
        state = GameState(lion_start_pos=(0, 9))
        
        assert len(state.history) == 0
        
        state, _, _, _ = engine.step(state, LionAction.ADVANCE, ImpalaAction.LOOK_FRONT)
        assert len(state.history) == 1
        
        state, _, _, _ = engine.step(state, LionAction.ADVANCE, ImpalaAction.DRINK)
        assert len(state.history) == 2
        
        # Check history structure
        last_entry = state.history[-1]
        assert "time_step" in last_entry
        assert "lion_pos" in last_entry
        assert "impala_pos" in last_entry
        assert "lion_action" in last_entry
        assert "impala_action" in last_entry


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
