#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

import time

import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace) -> float:
    # Load digit dataset
    dataset = sklearn.datasets.load_digits()
    dataset.target = dataset.target % 2

    # If you want to learn about the dataset, you can print some information
    # about it using `print(dataset.DESCR)`.

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.

    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(dataset.data, dataset.target, 
    test_size=args.test_size, random_state=args.seed)

    # TODO: Create a pipeline, which
    # 1. performs sklearn.preprocessing.MinMaxScaler()
    # 2. performs sklearn.preprocessing.PolynomialFeatures()
    # 3. performs sklearn.linear_model.LogisticRegression(random_state=args.seed)
    #
    # Then, using sklearn.model_selection.StratifiedKFold(5), evaluate crossvalidated
    # train performance of all combinations of the following parameters:
    # - polynomial degree: 1, 2
    # - LogisticRegression regularization C: 0.01, 1, 100
    # - LogisticRegression solver: lbfgs, sag
    #
    # For the best combination of parameters, compute the test set accuracy.
    #
    # The easiest way is to use `sklearn.model_selection.GridSearchCV`.

    # poly_deg = 2           # [1, 2]
    # solver_log_r = 'lbfgs' # ['lbfgs', 'sag']
    # C_log_r = 1.0          # [ 0.01, 1.0, 100.0]
    pipe = sklearn.pipeline.Pipeline([('minmaxsc', sklearn.preprocessing.MinMaxScaler() ),
                                      ('polynomial',    sklearn.preprocessing.PolynomialFeatures() ),
                                      ('lr',  sklearn.linear_model.LogisticRegression(random_state=args.seed) )
                                     ])   # verbose=True

    # sklearn.model_selection.GridSearchCV(estimator, param_grid, *, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=nan, return_train_score=False)[source]
    # estimator = pipe
    # param_grid
    parameters = {
        "polynomial__degree": (1, 2),
        "lr__solver": ('lbfgs', 'sag'),
        "lr__C": ( 0.01, 1.0, 100.0),
    }

    t0 = time.time()

    gs = sklearn.model_selection.GridSearchCV(pipe, parameters, cv=5, refit=True, verbose=0)  # refit=True (default), n_jobs = 4.... pocet vlaken

    # fit and prediction
    gs.fit(train_data, train_target)
    # print(gs.get_params())
    y_pred = gs.predict(test_data)

    test_accuracy = gs.score(test_data, test_target)
    # print(sklearn.metrics.classification_report(test_target, y_pred))
    # test_accuracy = gs.score_samples(test_data)

    # test_accuracy = None

    # print('{}s'.format(time.time()-t0))

    return test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(args)
    print("Test accuracy: {:.2f}".format(100 * test_accuracy))
