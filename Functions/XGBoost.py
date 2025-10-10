import os
import pickle

class XGBoost():
    @staticmethod
    def XGM():
        model_path = os.path.join(os.path.dirname(__file__), "../model/xgb_model.pkl")
        with open(model_path, "rb") as f:
            clf = pickle.load(f)
        return clf
