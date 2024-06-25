#!/usr/bin/env python3
#
# Team members' IDs:
# 572e4c43-e678-11e9-9ce9-00505601122b  (Jaro Luknis)
# fa2094b3-2eab-11ec-986f-f39926f24a9c  (Jan Zubáč)
# a6ef6adf-e5c9-11e9-9ce9-00505601122b  (Ondřej Varga)
#
#
#  grad boost classifier loss 0.0031
#


#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys

import numpy as np
import pandas as pd

# my imports
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
import sklearn.pipeline

from sklearn.ensemble import GradientBoostingClassifier


class Dataset:
    CLASSES = ["sitting", "sittingdown", "standing", "standingup", "walking"]

    def __init__(self,
                 name="human_activity_recognition.train.csv.xz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and if it contains column "class", split it to `targets`.
        self.data = pd.read_csv(name)
        if "class" in self.data:
            self.target = np.array([Dataset.CLASSES.index(target) for target in self.data["class"]], np.int32)
            self.data = self.data.drop("class", axis=1)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="human_activity_recognition_v14.model", type=str, help="Model path")


def plot_sample(data, data_target, ind=100):
    fig = plt.figure()
    ax1 = fig.add_subplot(511)
    ax2 = fig.add_subplot(512)
    ax3 = fig.add_subplot(513)
    ax4 = fig.add_subplot(514)
    ax5 = fig.add_subplot(515)
    ax1.plot(data.waist_x[:ind])
    ax1.plot(data.waist_y[:ind])
    ax1.plot(data.waist_z[:ind])
    ax1.title.set_text("waist")
    
    ax2.plot(data.thigh_x[:ind])
    ax2.plot(data.thigh_y[:ind])
    ax2.plot(data.thigh_z[:ind])
    ax2.title.set_text("thigh")
    ax3.plot(data.ankle_x[:ind])
    ax3.plot(data.ankle_y[:ind])
    ax3.plot(data.ankle_z[:ind])
    ax3.title.set_text("ankle")
    ax4.plot(data.arm_x[:ind])
    ax4.plot(data.arm_y[:ind])
    ax4.plot(data.arm_z[:ind])
    ax4.title.set_text("arm")
    ax5.plot(data_target[:ind])
    ax5.title.set_text("activity")
    plt.tight_layout()
    plt.show()

    return

def plot_correlations(data):
    df = pd.DataFrame(np.array(data))
    plt.matshow(df.corr())
    plt.show()
    return


def main(args: argparse.Namespace):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # print("train.data",train.data.shape)
        


        pipe = sklearn.pipeline.Pipeline([
                # ('poly_features', sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)),
                # ('mlp', MLPClassifier(max_iter=1000, hidden_layer_sizes=(100,50), verbose=True, tol=0.000000001,  n_iter_no_change=10) )
                # ('gb', GradientBoostingClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, validation_fraction=0.1) ),
                ('gb', GradientBoostingClassifier(max_depth=6, learning_rate=0.1, n_estimators=500, validation_fraction=0.1, verbose = 10,
                random_state= np.random.randint(0,200) ) ),
                                 ])

        model = pipe.fit(train.data, train.target)



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
        predictions = model.predict(test.data)
        
        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
