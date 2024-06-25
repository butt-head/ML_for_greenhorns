#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import urllib.request
import numpy as np

# my
import sklearn.pipeline
import sklearn.linear_model
import sklearn.compose
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)

class Dataset:
    """Thyroid Dataset.

    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features

    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """
    def __init__(self,
                 name="thyroid_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name))
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="thyroid_competition_v3.model", type=str, help="Model path")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")


def main(args: argparse.Namespace):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        cols_bin  = [j for j in range(15)]
        cols_real = [15, 16, 17, 18, 19, 20]

        # print(train.data[1,cols_bin])
        # print(train.data[1,cols_real])

        # TODO: Train a model on the given dataset and store it in `model`.
        # pipe = sklearn.pipeline.Pipeline([('minmaxsc', sklearn.preprocessing.MinMaxScaler() ),
        #                           ('polynomial',    sklearn.preprocessing.PolynomialFeatures() ),
        #                           ('lr',  sklearn.linear_model.LogisticRegression(random_state=args.seed) )
        #                          ])   # verbose=Tr

        # test/train splitting
        train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        train.data, train.target, 
        test_size=args.test_size, 
        random_state=args.seed)

        pipe = sklearn.pipeline.Pipeline([
                                ('col-trans', sklearn.compose.ColumnTransformer([
                                            ('quant', sklearn.preprocessing.QuantileTransformer(), cols_real)])),
                                ('poly_features', sklearn.preprocessing.PolynomialFeatures(3, include_bias=False)),
                                ('lr',  sklearn.linear_model.LogisticRegression(random_state=args.seed) )
                                 ])   # verbose=Tr

        # ('poly_features', sklearn.preprocessing.PolynomialFeatures(2, include_bias=False))

        parameters = {
        "poly_features__degree": (1, 2, 3, 4),
        "lr__solver": ('newton-cg','lbfgs', 'liblinear', 'sag', 'saga'),
        "lr__C": ( 0.01, 0.1, 1.0, 10, 100.0, 1000),
        #"lr__cv": (5, 10, 20)
        }

        # parameters = {
        # "poly_features__degree": (1, 2, 3),
        # "lr__solver": ('newton-cg', 'liblinear', 'saga'),
        # "lr__C": ( 0.1, 1.0, 10),
        # #"lr__cv": (5, 10, 20)
        # }

        gs = sklearn.model_selection.GridSearchCV(pipe, parameters, cv=8, refit=True, verbose=2) 
        
        # pipe.fit(train.data, train.target)
        # pipe.fit(train_data, train_target)

        # model = None
        # model = pipe.fit(train_data, train_target)
        # model = pipe.fit(train.data, train.target)

        # model = gs.fit(train_data, train_target)
        model = gs.fit(train.data, train.target)

        # print(model.score(test_data, test_target))

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        # predictions = None
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
