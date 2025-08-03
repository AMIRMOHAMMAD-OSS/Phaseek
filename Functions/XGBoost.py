import pickle
class XGBoost():
  def XGM():
    with open('model/xgb_model.pkl', 'rb') as f:
      clf = pickle.load(f)
    return clf
