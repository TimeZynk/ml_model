import pickle


class Model(object):
    def __init__(self, pickled_model):
        with open(pickled_model, "rb") as f:
            self.MinMaxScaler, self.best_estimator, self.LabelEncoder = pickle.load(f)
