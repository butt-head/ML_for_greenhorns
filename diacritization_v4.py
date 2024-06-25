import argparse
import lzma
import pickle
import os
import sys
import urllib.request


import numpy as np
from numpy.lib.function_base import vectorize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import sklearn.pipeline
import re

import joblib

import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)


class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)
            urllib.request.urlretrieve(url + name.replace(".txt", ".LICENSE"), filename=name.replace(".txt", ".LICENSE"))

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization_ngram1_5_50iter_prot_m1.model", type=str, help="Model path")
#parser.add_argument("--model_path", default="diacritization_ngram1_5_50iter.model", type=str, help="Model path")
#parser.add_argument("--model_path2", default="diacritization_ngram1_5_50iter_small.model", type=str, help="Model path")
#parser.add_argument("--model_path", default="diacritization_ngram1_5_50iter_small.model", type=str, help="Model path")
parser.add_argument("--dataset_name", default="fiction-train.txt", type=str, help="Dataset path")


def accuracy(gold: str, system: str): #-> float:
    assert isinstance(gold, str) and isinstance(system, str), "The gold and system outputs must be strings"

    gold, system = gold.split(), system.split()
    assert len(gold) == len(system), "The gold and system outputs must have same number of words: {} vs {}.".format(len(gold), len(system))

    words, correct = 0, 0
    for gold_token, system_token in zip(gold, system):
        words += 1
        correct += gold_token == system_token

    return correct / words


def main(args: argparse.Namespace):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset(name=args.dataset_name)

        ## split of train.data
        sentences_train = train.data.split("\n")
        sentences_train_target = train.target.split("\n")

        pipe = sklearn.pipeline.Pipeline([
                     ( 'count-vectorizer',  CountVectorizer(analyzer='char_wb', ngram_range=(1, 5), lowercase = False )   ),
                    ('mlp', MLPClassifier(max_iter=50, hidden_layer_sizes=(100,200), verbose=True) )
                                 ])
        


        model = pipe.fit(sentences_train, sentences_train_target)
        sentences_predict = model.predict(sentences_train)

        print("score:", model.score(sentences_train, sentences_train_target))
        try:
            print("accuracy:", accuracy("\n".join(sentences_predict), "\n".join(sentences_train_target)))
        except AssertionError:
            print('assert_err')

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file, protocol=-1)

        return model, "\n".join(sentences_predict), "\n".join(sentences_train_target)

    else:
        # Use the model and return test set predictions.
    

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # with lzma.open(args.model_path2, "wb") as model_file:
        #     joblib.dump(model, model_file, protocol=-1)

        # with lzma.open(args.model_path2, "rb") as model_file:
        #     model_comp = lzma.compress(model)
        #     joblib.dump(model_comp, model_file, protocol=-1)


        # with lzma.open(args.model_path2, "wb") as model_file:
        #     pickle.dump(model, model_file, protocol=-1)

        


        # TODO: Generate `predictions` with the test set predictions. Specifically,
        # produce a diacritized `str` with exactly the same number of words as `test.data`.

        test = Dataset(args.predict)

        sentences_test = test.data.split("\n")
        predictions = model.predict(sentences_test)
        predictions = "\n".join(predictions)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)