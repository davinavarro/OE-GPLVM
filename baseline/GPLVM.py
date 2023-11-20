from utils.myutils import Utils
import numpy as np

class GPLVM:
    def __init__(self, seed, model_name, tune=False):
        self.seed = seed
        self.utils = Utils()

        self.model_name = model_name
        self.model_dict = {
            "IForest": IForest,
        }

        self.tune = tune

    def grid_hp(self, model_name):
          return None
    
    def grid_search(self, X_train, y_train, ratio=None):
        return None
    
    def fit(self, X_train, y_train, ratio=None):
        return self

    # from pyod: for consistency, outliers are assigned with larger anomaly scores
    def predict_score(self, X):
        score = self.model.decision_function(X)
        return score
