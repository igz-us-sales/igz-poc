from typing import List

import numpy as np
import storey
from cloudpickle import load

import mlrun
from mlrun.serving.routers import ParallelRun

class ClassifierModel(mlrun.serving.V2ModelServer):
    def load(self):
        """load and initialize the model and/or other elements"""
        model_file, extra_data = self.get_model(".pkl")
        self.model = load(open(model_file, "rb"))

    def predict(self, body: dict) -> List:
        """Generate model predictions from sample."""
        feats = np.asarray(body["inputs"])
        result: np.ndarray = self.model.predict(feats)
        return result.tolist()

class ParallelRouter(ParallelRun):
        
    def merger(self, body, results):
        """Merging logic
        input the event body and a dict of route results and returns a dict with merged results
        
        Incoming results dictionary will look like:
        results = {
            'champion': {'id': '12f9d8b5317b444fa3773ff6d96d3ebe', 'model_name': 'sepal_length_cm', 'outputs': [0, 2]},
            'challenger': {'id': '12f9d8b5317b444fa3773ff6d96d3ebe', 'model_name': 'petal_width_cm', 'outputs': [0, 2]}
        }
        
        Merged result will look like:
        merged = {
            'champion': [0, 2],
            'challenger': [0, 2]
        }
        """
        print()
        print("Merger in:", results)
        merged = {result["model_name"] : result["outputs"] for result in results.values()}
        print("Merger out:", merged)
        return merged
