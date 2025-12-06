import pickle
import os

class PickleStorage:
    @staticmethod
    def save(data: object, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load(filepath: str) -> object:
        with open(filepath, "rb") as f:
            return pickle.load(f)
