import argparse
import os
import sys
from time import time

from DataLoader import DataLoader
from FingerprintIdentifier import FingerprintIdentifier


class Timer(object):
    """
    with Timer("name"): to make timing easier
    """

    def __init__(self, name=None, output=True):
        self.name = name
        self.output = output
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, *args):
        self.end = time()
        self.elapsed = self.end - self.start
        if self.output:
            print(self)

    def __str__(self):
        return "{}: {:0.6f} seconds".format(self.name, self.elapsed) if self.name else "{:0.6f} seconds".format(self.elapsed)


def test(args):
    useTimer = args.time
    loader = DataLoader()
    with Timer('Load Unfiltered Data', useTimer):
        loader.load_samples()

    identifier = FingerprintIdentifier(loader.get(start=1))
    with Timer('SVM Train', useTimer):
        identifier.train()

    test = loader.get(start=0, end=1)
    correct = 0
    incorrect = 0
    total = 0

    for k, vs in test.items():
        for v in vs:
            prediction = identifier.identify([v])
            # print("Actual ", k, " Predicted ", prediction)
            if k == prediction:
                correct += 1
            else:
                incorrect += 1
            total += 1

    print("Unfiltered Data: {:.3f}% correct {:.3f}% incorrect".format(correct/total * 100, incorrect/total * 100))

    del loader

    loader = DataLoader(use_gabor=True)
    with Timer('Load Gabor Data', useTimer):
        loader.load_samples()

    identifier = FingerprintIdentifier(loader.get(start=1))
    with Timer('Train Gabor SVM', useTimer):
        identifier.train()

    test = loader.get(start=0, end=1)
    correct = 0
    incorrect = 0
    total = 0
    for k, vs in test.items():
        for v in vs:
            prediction = identifier.identify([v])
            # print("Actual ", k, " Predicted ", prediction)
            if k == prediction:
                correct += 1
            else:
                incorrect += 1
            total += 1

    print("Gabor Filter: {:.3f}% correct {:.3f}% incorrect".format(correct/total * 100, incorrect/total * 100))

    del loader

    loader = DataLoader(pca_components=3)
    with Timer('Load PCA Data', useTimer):
        loader.load_samples()

    identifier = FingerprintIdentifier(loader.get(start=1))
    with Timer('Train PCA', useTimer):
        identifier.train()
    test = loader.get(start=0, end=1)
    correct = 0
    incorrect = 0
    total = 0
    for k, vs in test.items():
        for v in vs:
            prediction = identifier.identify([v])
            # print("Actual ", k, " Predicted ", prediction)
            if k == prediction:
                correct += 1
            else:
                incorrect += 1
            total += 1

    print("PCA: {:.3f}% correct {:.3f}% incorrect".format(correct/total * 100, incorrect/total * 100))


def main(args):
    training = 'training'
    testing = args.filename
    if len(args.filename) > 1:
        first = args.filename[0]
        if os.path.isdir(first):
            training = first
            testing = args.filename[1:]

    loader = DataLoader()
    loader.load_samples(folder=training)

    identifier = FingerprintIdentifier(loader.get())
    identifier.train()
    for filename in testing:
        f = loader.load_file(filename)
        prediction = identifier.identify(f)

        print("Unfiltered Predicted ", prediction, ' for ', filename)

    del loader

    loader = DataLoader(use_gabor=True)
    loader.load_samples(folder=training)

    identifier = FingerprintIdentifier(loader.get())
    identifier.train()
    for filename in testing:
        f = loader.load_file(filename)
        prediction = identifier.identify(f)

        print("Gabor Filter Predicted ", prediction, ' for ', filename)

    del loader

    loader = DataLoader(pca_components=3)
    loader.load_samples(folder=training)

    identifier = FingerprintIdentifier(loader.get())
    identifier.train()
    for filename in testing:
        f = loader.load_file(filename)
        prediction = identifier.identify(f)

        print("PCA Predicted ", prediction, ' for ', filename)


def parseArgs(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs='+')
    parser.add_argument('-t', '--time', action='store_true')
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parseArgs(sys.argv[1:])
    main(args)
