import numpy as np
import sklearn


class FingerprintIdentifier(object):

    def __init__(self, data):
        self.data = data
        self.models = [sklearn.svm.SVC() for _ in range(21)]
        self.train()

    def model(self, subject):
        return self.models[subject-1]

    def train(self):
        for subject in self.data.keys():
            model = self.model(subject)
            x = self.data[subject]
            y = [1] * len(x)
            for k, v in self.data.items():
                if k == subject:
                    continue
                x.extend(v)
                y.extend([0] * len(v))
            model.fit(x, y)

    def identify(self, data):
        predictions = np.array([model.predict(data) for model in self.models])
        return predictions.argmax()
