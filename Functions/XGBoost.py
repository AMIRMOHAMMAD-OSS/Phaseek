import pickle
class XGBoost():
  def XGM():
    with open('../xgb_model (1).pkl', 'rb') as f:
      clf = pickle.load(f)
    return clf
