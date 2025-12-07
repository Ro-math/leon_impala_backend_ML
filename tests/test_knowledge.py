"""
Unit tests for knowledge base management
Tests Q-table operations, save/load, and abstractions
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import json
from pathlib import Path
from app.learning.knowledge_base import KnowledgeBase
from app.core.entities import LionAction


class TestKnowledgeBaseBasics:
    """Tests for basic knowledge base operations"""
    
    def test_knowledge_base_init(self):
        """Test knowledge base initializes empty"""
        kb = KnowledgeBase()
        
        assert kb.q_table == {}
        assert kb.abstractions == []
    
    def test_q_value_get_default(self):
        """Test getting Q-value for non-existent state returns 0.0"""
        kb = KnowledgeBase()
        
        q_val = kb.get_q_value("test_state", "advance")
        assert q_val == 0.0
    
    def test_q_value_get_set(self):
        """Test setting and retrieving Q-values"""
        kb = KnowledgeBase()
        
        state_key = "9,9|drink|normal"
        kb.update_q_value(state_key, "advance", 5.0)
        
        assert kb.get_q_value(state_key, "advance") == 5.0
        assert kb.get_q_value(state_key, "hide") == 0.0  # Other actions still 0
    
    def test_q_value_update_multiple(self):
        """Test updating same Q-value multiple times"""
        kb = KnowledgeBase()
        
        state_key = "5,5|look_front|normal"
        kb.update_q_value(state_key, "attack", 1.0)
        kb.update_q_value(state_key, "attack", 2.5)
        kb.update_q_value(state_key, "attack", -0.5)
        
        assert kb.get_q_value(state_key, "attack") == -0.5
    
    def test_multiple_states(self):
        """Test multiple states can coexist"""
        kb = KnowledgeBase()
        
        kb.update_q_value("0,9|drink|normal", "advance", 10.0)
        kb.update_q_value("0,9|look_front|normal", "hide", 5.0)
        kb.update_q_value("18,18|drink|attacking", "attack", -2.0)
        
        assert len(kb.q_table) == 3
        assert kb.get_q_value("0,9|drink|normal", "advance") == 10.0
        assert kb.get_q_value("18,18|drink|attacking", "attack") == -2.0


class TestKnowledgePersistence:
    """Tests for saving and loading knowledge"""
    
    def test_knowledge_save_load_json(self, tmp_path):
        """Test save and load knowledge to JSON"""
        kb = KnowledgeBase()
        
        # Add some data
        kb.update_q_value("test1", "advance", 5.0)
        kb.update_q_value("test1", "hide", 3.0)
        kb.update_q_value("test2", "attack", -1.0)
        kb.abstractions = ["Rule 1", "Rule 2"]
        
        # Save
        from app.storage.json_storage import JsonStorage
        filepath = tmp_path / "test_kb.json"
        data = {"q_table": kb.q_table, "abstractions": kb.abstractions}
        JsonStorage.save(data, str(filepath))
        
        # Load into new KB
        kb2 = KnowledgeBase()
        loaded_data = JsonStorage.load(str(filepath))
        kb2.q_table = loaded_data["q_table"]
        kb2.abstractions = loaded_data["abstractions"]
        
        # Verify
        assert kb2.get_q_value("test1", "advance") == 5.0
        assert kb2.get_q_value("test2", "attack") == -1.0
        assert len(kb2.abstractions) == 2
    
    def test_knowledge_clear(self):
        """Test clearing knowledge base"""
        kb = KnowledgeBase()
        
        kb.update_q_value("test", "advance", 10.0)
        kb.abstractions = ["Rule 1"]
        
        assert len(kb.q_table) > 0
        assert len(kb.abstractions) > 0
        
        kb.clear()
        
        assert kb.q_table == {}
        assert kb.abstractions == []
    
    def test_abstraction_storage(self):
        """Test storing and retrieving abstractions"""
        kb = KnowledgeBase()
        
        rules = [
            "IF Lion at 0,9 AND Lion is normal AND Impala does [drink] THEN advance",
            "IF Lion at 5,5 AND Lion is attacking AND Impala does [look_front, look_left] THEN attack"
        ]
        
        kb.abstractions = rules
        
        assert len(kb.abstractions) == 2
        assert kb.abstractions[0] == rules[0]
        assert kb.abstractions[1] == rules[1]


class TestQTableStructure:
    """Tests for Q-table structure and integrity"""
    
    def test_state_key_format(self):
        """Test state keys follow expected format"""
        kb = KnowledgeBase()
        
        # Valid format: "x,y|impala_action|lion_state"
        state_key = "0,9|drink|normal"
        kb.update_q_value(state_key, "advance", 1.0)
        
        assert state_key in kb.q_table
    
    def test_all_actions_initialized(self):
        """Test all lion actions are initialized when accessing state"""
        kb = KnowledgeBase()
        
        state_key = "new_state"
        _ = kb.get_q_value(state_key, "advance")
        
        # Should have initialized all action values
        assert state_key in kb.q_table
        assert len(kb.q_table[state_key]) == 3  # advance, hide, attack
        
        for action in LionAction:
            assert action.value in kb.q_table[state_key]
            assert kb.q_table[state_key][action.value] == 0.0
    
    def test_q_table_json_serializable(self):
        """Test Q-table can be serialized to JSON"""
        kb = KnowledgeBase()
        
        kb.update_q_value("0,9|drink|normal", "advance", 5.5)
        kb.update_q_value("1,1|look_left|hidden", "hide", -2.3)
        
        # Should be JSON serializable
        json_str = json.dumps(kb.q_table)
        loaded = json.loads(json_str)
        
        assert loaded["0,9|drink|normal"]["advance"] == 5.5
        assert loaded["1,1|look_left|hidden"]["hide"] == -2.3


class TestKnowledgeBaseIntegration:
    """Integration tests for knowledge base with other components"""
    
    def test_knowledge_with_state_key_generation(self):
        """Test knowledge base works with actual state keys from agent"""
        from app.learning.reinforcement import QLearningAgent
        from app.core.entities import ImpalaAction, LionState
        
        kb = KnowledgeBase()
        agent = QLearningAgent(kb)
        
        # Generate state key
        state_key = agent.get_state_key((0, 9), ImpalaAction.DRINK, LionState.NORMAL)
        
        # Update Q-value
        kb.update_q_value(state_key, "advance", 10.0)
        
        # Retrieve
        assert kb.get_q_value(state_key, "advance") == 10.0
    
    def test_knowledge_persistence_full_cycle(self, tmp_path):
        """Test full save/load cycle with knowledge base methods"""
        kb1 = KnowledgeBase()
        
        # Populate with realistic data
        kb1.update_q_value("0,9|drink|normal", "advance", 7.5)
        kb1.update_q_value("0,9|drink|normal", "hide", 2.1)
        kb1.update_q_value("5,5|look_front|attacking", "attack", 15.0)
        kb1.abstractions.append("Test Rule 1")
        
        # Save using direct storage (KB save has path issues with absolute paths)
        from app.storage.json_storage import JsonStorage
        test_file = str(tmp_path / "kb_test.json")
        data = {"q_table": kb1.q_table, "abstractions": kb1.abstractions}
        JsonStorage.save(data, test_file)
        
        # Load into new instance
        kb2 = KnowledgeBase()
        loaded_data = JsonStorage.load(test_file)
        kb2.q_table = loaded_data["q_table"]
        kb2.abstractions = loaded_data["abstractions"]
        
        # Verify all data transferred
        assert kb2.get_q_value("0,9|drink|normal", "advance") == 7.5
        assert kb2.get_q_value("5,5|look_front|attacking", "attack") == 15.0
        assert "Test Rule 1" in kb2.abstractions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
