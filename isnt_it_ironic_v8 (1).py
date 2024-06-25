#!/usr/bin/env python3
#
# Team members' IDs:
# 572e4c43-e678-11e9-9ce9-00505601122b  (Jaro Luknis)
# fa2094b3-2eab-11ec-986f-f39926f24a9c  (Jan Zubáč)
# a6ef6adf-e5c9-11e9-9ce9-00505601122b  (Ondřej Varga)
# 
# Iteration 750, loss = 0.00023604
# Training loss did not improve more than tol=0.0000002 for 10 consecutive epochs. Stopping.
# over baseline 58.5%
#


import argparse
import lzma
import pickle
import os
import urllib.request
import sys

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import sklearn.pipeline
import re


class Dataset:
    def __init__(self,
                 name="isnt_it_ironic.train.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)
            urllib.request.urlretrieve(url + name.replace(".txt", ".LICENSE"), filename=name.replace(".txt", ".LICENSE"))

        # Load the dataset and split it into `data` and `target`.
        self.data = []
        self.target = []

        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            for line in dataset_file:
                label, text = line.rstrip("\n").split("\t")
                self.data.append(text)
                self.target.append(int(label))
        self.target = np.array(self.target, np.int32)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="isnt_it_ironic_v8.model", type=str, help="Model path")


def main(args: argparse.Namespace):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()


        # print(train.data)
        # print(train.target)
        # print(len(train.target))


        # sentences_train = train.data.split("\n")
        # sentences_train_target = train.target.split("\n")

        pipe = sklearn.pipeline.Pipeline([
                     ( 'count-vectorizer',  CountVectorizer(analyzer='word', ngram_range=(1, 3), lowercase = True )   ),
                     ('mlp', MLPClassifier(max_iter=1000, hidden_layer_sizes=(11,2), verbose=True, tol=0.0000002) )
                                 ])

        model = pipe.fit(train.data, train.target)

        

        # TODO: Train a model on the given dataset and store it in `model`.
        # model = None

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions, either
        # as a Python list or a NumPy array.
        # predictions = None
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
