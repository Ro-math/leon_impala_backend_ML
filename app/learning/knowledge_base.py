import json
import pickle
from typing import Dict, List, Any, Tuple
from app.core.entities import LionAction, ImpalaAction

class KnowledgeBase:
    def __init__(self):
        # Q-Table: Key = (LionPos, ImpalaAction, LionState), Value = {Action: Q-Value}
        # We need a string representation for the key to serialize easily to JSON.
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.abstractions: List[str] = []

    def get_q_value(self, state_key: str, action: str) -> float:
        if state_key not in self.q_table:
            self.q_table[state_key] = {a.value: 0.0 for a in LionAction}
        return self.q_table[state_key].get(action, 0.0)

    def update_q_value(self, state_key: str, action: str, value: float):
        if state_key not in self.q_table:
            self.q_table[state_key] = {a.value: 0.0 for a in LionAction}
        self.q_table[state_key][action] = value

    def save(self, filename: str, format: str = "json"):
        filepath = f"data/knowledge/{filename}"
        data = {"q_table": self.q_table, "abstractions": self.abstractions}
        
        if format == "json":
            from app.storage.json_storage import JsonStorage
            JsonStorage.save(data, filepath + ".json")
        elif format == "pickle":
            from app.storage.pickle_storage import PickleStorage
            PickleStorage.save(data, filepath + ".pkl")

    def load(self, filename: str):
        filepath_base = f"data/knowledge/{filename}"
        
        try:
            # Try JSON first
            from app.storage.json_storage import JsonStorage
            data = JsonStorage.load(filepath_base + ".json")
            self.q_table = data["q_table"]
            self.abstractions = data.get("abstractions", [])
        except FileNotFoundError:
            try:
                # Try Pickle
                from app.storage.pickle_storage import PickleStorage
                data = PickleStorage.load(filepath_base + ".pkl")
                self.q_table = data["q_table"]
                self.abstractions = data.get("abstractions", [])
            except FileNotFoundError:
                print(f"Knowledge file {filename} not found.")

    def clear(self):
        self.q_table = {}
        self.abstractions = []
