import sys
import os
sys.path.append(os.getcwd())

from app.core.entities import Lion, Impala, GameMap, LionAction, ImpalaAction, LionState
from app.core.game_engine import GameEngine, GameState
from app.core.vision_calculator import VisionCalculator

def test_vision():
    print("Testing Vision...")
    vc = VisionCalculator()
    
    # Lion at (0, 9) - West Middle. Impala at (9,9).
    # Impala looks Front (West). Should see Lion.
    lion_pos = (0, 9)
    visible = vc.is_lion_visible(lion_pos, LionState.NORMAL, ImpalaAction.LOOK_FRONT)
    print(f"Lion at (0,9), Impala Look Front: Visible={visible} (Expected: True)")
    
    # Impala looks Left (North). Should NOT see Lion.
    visible = vc.is_lion_visible(lion_pos, LionState.NORMAL, ImpalaAction.LOOK_LEFT)
    print(f"Lion at (0,9), Impala Look Left: Visible={visible} (Expected: False)")
    
    # Lion Hidden
    visible = vc.is_lion_visible(lion_pos, LionState.HIDDEN, ImpalaAction.LOOK_FRONT)
    print(f"Lion at (0,9) HIDDEN, Impala Look Front: Visible={visible} (Expected: False)")

def test_game_engine():
    print("\nTesting Game Engine...")
    engine = GameEngine()
    state = GameState(lion_start_pos=(0, 9))
    
    # Step 1: Lion Advances, Impala Looks Right (South).
    # Lion at (0,9). Impala looks South. Lion not visible.
    # Lion moves to (1, 9).
    next_state, reward, done, info = engine.step(state, LionAction.ADVANCE, ImpalaAction.LOOK_RIGHT)
    print(f"Step 1: Lion Pos={next_state.lion.position}, Info={info}, Done={done}")
    
    # Step 2: Lion Attacks! Impala Looks Front.
    # Lion at (1,9). Impala looks Front (West). Sees Lion! Flee Triggered.
    # Lion moves 2 steps -> (3, 9).
    next_state, reward, done, info = engine.step(next_state, LionAction.ATTACK, ImpalaAction.LOOK_FRONT)
    print(f"Step 2: Lion Pos={next_state.lion.position}, Info={info}, Done={done}")
    print(f"Impala State: {next_state.impala.state}")
    
    # Test History
    print(f"History Length: {len(next_state.history)} (Expected: 2)")

if __name__ == "__main__":
    test_vision()
    test_game_engine()
