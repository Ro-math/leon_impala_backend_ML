import json
import os

class JsonStorage:
    @staticmethod
    def save(data: dict, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(filepath: str) -> dict:
        with open(filepath, "r") as f:
            return json.load(f)
