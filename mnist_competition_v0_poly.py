#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import urllib.request

import numpy as np

# machine learning
from sklearn.metrics import f1_score
import gc
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import sklearn.svm

# my
import sklearn.pipeline
import sklearn.linear_model
import sklearn.compose
import warnings
import sklearn.exceptions
import sys
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)

class Dataset:
    """MNIST Dataset.

    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in range 0-255, the targets are numbers 0-9.
    """
    def __init__(self,
                 name="mnist.train.npz",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name))
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset, i.e., `data` and optionally `target`.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value[:data_size])
        self.data = self.data.reshape([-1, 28*28]).astype(np.float)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="mnist_competition_svc_poly.model", type=str, help="Model path")
parser.add_argument("--test_size", default=0.85, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")


def main(args: argparse.Namespace):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # TODO: Train a model on the given dataset and store it in `model`.
        model = None


        # test/train splitting
        train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        train.data, train.target, 
        test_size=args.test_size, 
        random_state=args.seed)


        # model = make_pipeline(StandardScaler(),
        #                       SVC(kernel=kernel,
        #                           probability=True,
        #                           random_state=2021)
        #   )

        pipe = sklearn.pipeline.Pipeline([
                                ('std',  StandardScaler() ),
                                # ('poly_features', sklearn.preprocessing.PolynomialFeatures(3, include_bias=False)),
                                ('svc', SVC(probability=True, random_state=2021) )
                                 ])   # verbose=Tr
        # If you trained one or more MLPs, you can use the following code
        # to compress it significantly (approximately 12 times). The snippet
        # assumes the trained MLPClassifier is in `mlp` variable.
        # mlp._optimizer = None
        # for i in range(len(mlp.coefs_)): mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
        # for i in range(len(mlp.intercepts_)): mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)

        parameters = {
        #"svc__kernel": ('linear','poly', 'rbf', 'sigmoid'),
        "svc__C": [10],
        "svc__kernel": ['poly'],
        #"svc__kernel": ('poly', 'rbf', 'sigmoid'),
        #"svc__gamma": (0.5, 1.0, 5, 10),
        # "svc__C": [0.01],

        # "svc__C": [0.1],
        #"lr__cv": (5, 10, 20)
        }

        # logs to file
        old_stdout = sys.stdout
        log_file = open("message.log","a")
        sys.stdout = log_file


        # gs = sklearn.model_selection.GridSearchCV(pipe, parameters, cv=2, refit=True, verbose=5)
        gs = sklearn.model_selection.GridSearchCV(pipe, parameters, cv=3, refit=True, verbose=10, n_jobs = 3,
        return_train_score=True)

        # back to old stdout
        sys.stdout = old_stdout
        log_file.close()


        # model = gs.fit(train_data, train_target)

        # print(model.score(train_data, train_target))

        model = gs.fit(train.data, train.target)
        print(model.score(train.data, train.target))
        # print(model.cv_results_ )
        


        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
