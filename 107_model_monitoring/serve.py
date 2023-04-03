from typing import List

import numpy as np
import storey
from cloudpickle import load
import random

import mlrun
from mlrun.serving.routers import ParallelRun

class ClassifierModel(mlrun.serving.V2ModelServer):
    
    def load(self):
        """load and initialize the model and/or other elements"""
        model_file, extra_data = self.get_model(".pkl")
        self.model = load(open(model_file, "rb"))
        
    def validate(self, request, operation):
        """Removed default validation"""
        return request
        
    def preprocess(self, request: dict, operation) -> dict:
        """preprocess the event body before validate and action"""
        request["inputs"] = [list(request.values())]
        return request

    def predict(self, body: dict) -> List:
        """Generate model predictions from sample."""
        feats = np.asarray(body["inputs"])
        result: np.ndarray = self.model.predict(feats)
            
        #### Custom KPI
        self.set_metric("my_kpi", random.uniform(1, 10))
        ####
        
        return result.tolist()

def format_prediction(event):
    CLASS_MAPPINGS = {0: "setosa", 1 : "versicolor", 2: "virginica"}

    return {
        "model_name" : event["model_name"],
        "prediction" : CLASS_MAPPINGS[event["outputs"][0]]
    }
