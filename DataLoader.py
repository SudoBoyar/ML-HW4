import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import convolve
from scipy.misc import imread
from skimage.filters import gabor_kernel
from sklearn.decomposition import PCA

import os


class DataLoader(object):

    def __init__(self, final_width=65, final_height=105, pca_components=None, use_gabor=False):
        height_start = int((255 - final_height) / 2)
        height_end = height_start + final_height
        width_start = int((255 - final_width) / 2)
        width_end = width_start + final_width
        self.resize = (slice(height_start, height_end), slice(width_start, width_end))

        self.pca_components = pca_components
        self.pca_model = PCA(n_components=self.pca_components)

        self.use_gabor = use_gabor
        self.gabor_kernels = None

        self.samples = {}

    def load_file(self, filename):
        sample = self.preprocess(imread(filename))

        if self.pca_components is not None:
            sample = self.pca_model.transform([sample])
            sample = sample.ravel()

        return [sample]

    def load_samples(self, folder='training'):
        files = sorted(os.listdir(folder))
        for filename in files:
            # print("loading ", filename)
            subject, sample_num = filename.split('.')[0].split('_')
            subject = int(subject)

            sample = self.preprocess(imread(os.path.join(folder, filename)))

            if subject not in self.samples:
                self.samples[subject] = []
            self.samples[subject].append(sample)

        if self.pca_components is not None:
            self.pca()

    def get(self, subject=None, start=0, end=None):
        if subject is None:
            # Get all
            return {k: v[start:end if end is not None else len(v)] for k, v in self.samples.items()}
        elif isinstance(subject, (list, tuple)):
            # Get subset of subjects
            return {k: v[start:end if end is not None else len(v)] for k, v in self.samples.items() if k in subject}
        elif subject in self.samples:
            # Get single subject
            return self.samples[subject][start:end if end is not None else len(self.samples[subject])]
        else:
            return None

    def preprocess(self, sample):
        # Greyscale
        sample = sample.mean(axis=2)

        # Normalize
        sample /= 255
        sample -= sample.mean()

        # plt.figure()
        # plt.imshow(sample * 255)
        # plt.show()
        # input()

        # Crop
        sample = sample[self.resize]

        # plt.figure()
        # plt.imshow(sample)
        # plt.show()
        # input()

        if self.use_gabor:
            sample = self.gabor_filter(sample)
        else:
            # Flatten
            sample = sample.ravel()

        return sample

    def pca(self):
        x = []
        for k, samples in self.samples.items():
            x.extend(samples)

        self.pca_model.fit(x)

        for k in self.samples.keys():
            self.samples[k] = self.pca_model.transform(self.samples[k])

    def get_gabor_kernels(self):
        if self.gabor_kernels is not None:
            return self.gabor_kernels

        self.gabor_kernels = []
        for theta in range(16):
            theta = theta / 16. * np.pi
            for sigma in (1, 3):
                for frequency in (0.05, 0.15, 0.25):
                    kernel = gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma)
                    self.gabor_kernels.append(kernel)

                    # plt.figure()
                    # plt.imshow(kernel, interpolation='nearest')
                    # plt.show()
        return self.gabor_kernels

    def gabor_filter(self, sample):
        features = []
        # plt.figure()
        # plt.imshow(sample)
        for kernel in self.get_gabor_kernels():
            # Mode in {‘reflect’,’constant’,’nearest’,’mirror’, ‘wrap’}
            result = convolve(sample, np.real(kernel), mode='wrap')
            features.append(result.mean())
            features.append(result.std())
            features.append(result.var())

        #     img = np.sqrt(convolve(sample, np.real(kernel))**2 + convolve(sample, np.imag(kernel))**2)
        #     plt.figure()
        #     plt.imshow(img)
        # plt.show()
        # input()
        return np.array(features)