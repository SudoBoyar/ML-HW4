import argparse
import os
import sys

from DataLoader import DataLoader
from FingerprintIdentifier import FingerprintIdentifier


def test(args):
    loader = DataLoader()
    loader.load_samples()

    identifier = FingerprintIdentifier(loader.get(start=1))
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

    print("SVM: ", correct/total * 100, '% correct ', incorrect/total * 100, '% incorrect')

    del loader

    loader = DataLoader(use_gabor=True)
    loader.load_samples()

    identifier = FingerprintIdentifier(loader.get(start=1))
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

    print("Gabor Filter: ", correct/total * 100, '% correct ', incorrect/total * 100, '% incorrect')

    del loader

    loader = DataLoader(pca_components=1000)
    loader.load_samples()

    identifier = FingerprintIdentifier(loader.get(start=1))
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

    print("PCA: ", correct/total * 100, '% correct ', incorrect/total * 100, '% incorrect')


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
    for filename in testing:
        f = loader.load_file(filename)
        prediction = identifier.identify(f)

        print("Unfiltered Predicted ", prediction, ' for ', filename)

    del loader

    loader = DataLoader(use_gabor=True)
    loader.load_samples(folder=training)

    identifier = FingerprintIdentifier(loader.get())
    for filename in testing:
        f = loader.load_file(filename)
        prediction = identifier.identify(f)

        print("Gabor Filter Predicted ", prediction, ' for ', filename)

    del loader

    loader = DataLoader(pca_components=1000)
    loader.load_samples(folder=training)

    for filename in testing:
        f = loader.load_file(filename)
        identifier = FingerprintIdentifier(loader.get())
        prediction = identifier.identify(f)

        print("PCA Predicted ", prediction, ' for ', filename)


def parseArgs(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs='+')
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parseArgs(sys.argv[1:])
    main(args)
