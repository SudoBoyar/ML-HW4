import sklearn.svm


class FingerprintIdentifier(object):

    def __init__(self, data):
        self.data = data
        self.model = sklearn.svm.SVC(decision_function_shape='ovr')
        self.train()

    def train(self):
        subjStart = {}
        subjEnd = {}
        x = []
        y = []
        for subject in sorted(self.data.keys()):
            subjStart[subject] = len(x)
            x.extend(self.data[subject])
            subjEnd[subject] = len(x)
            y.extend([subject] * len(self.data[subject]))

        self.model.fit(x, y)

    def identify(self, data):
        return self.model.predict(data)[0]
