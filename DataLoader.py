import numpy as np
from scipy.misc import imread

import os


class DataLoader(object):

    def __init__(self):
        self.samples = {}

    def load(self, folder='training'):
        files = os.listdir(folder)
        for filename in files:
            subject, sample_num = filename.split('.')[0].split('_')

            sample = self.preprocess(imread(filename))

            if subject not in self.samples:
                self.samples[subject] = []
            self.samples[subject].append(sample)

    def get(self, subject=None, start=0, end=None):
        if subject is None:
            # Get all
            return {k: v[start:end if end is not None else len(v)] for k, v in self.samples}
        elif isinstance(subject, (list, tuple)):
            # Get subset of subjects
            return {k: v[start:end if end is not None else len(v)] for k, v in self.samples.items() if k in subject}
        elif subject in self.samples:
            # Get single subject
            return self.samples[subject][start:end if end is not None else len(self.samples[subject])]
        else:
            return None

    def preprocess(self, sample):
        # Convert to greyscale
        sample = sample.mean(axis=2)

        # Find rows and columns that have no fingerprint data
        nonzeroCols = sample.any(axis=0)
        nonzeroRows = sample.any(axis=1)

        removeRows = [i for i, nonzero in enumerate(nonzeroRows) if not nonzero]
        removeCols = [i for i, nonzero in enumerate(nonzeroCols) if not nonzero]

        # Remove the empty rows/columns
        sample = np.delete(sample, removeRows, 0)
        sample = np.delete(sample, removeCols, 1)

        # Convert all data to single value since the pressure and amount of ink affect the pixel value
        sample = sample.ravel()
        for i in range(sample.shape[0]):
            if sample[i] > 0:
                sample[i] = 255

        return sample
