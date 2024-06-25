#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.1, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace) -> float:
    # Load the Diabetes dataset
    dataset = sklearn.datasets.load_diabetes()

    # The input data are in `dataset.data`, targets are in `dataset.target`.

    # If you want to learn about the dataset, you can print some information
    # about it using `print(dataset.DESCR)`.

    # DONE TODO: Append a new feature to all input data, with value "1"

    # DONE TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.

    # DONE TODO: Solve the linear regression using the algorithm from the lecture,
    # explicitly computing the matrix inverse (using `np.linalg.inv`).

    # DONE TODO: Predict target values on the test set.

    # DONE TODO: Compute root mean square error on the test set predictions.

    data = dataset.data
    target = dataset.target

    # adding column with ones
    ones = np.ones((data.shape[0], 1))
    data = np.concatenate((data,ones),axis = 1)

    # splitting to test and train
    data_train, data_test, target_train, target_test = sklearn.model_selection.train_test_split(data, target, 
    test_size=args.test_size, random_state=args.seed)

    # weigths computation: w = (X.T * X)^(-1) X.T * t
    weights = np.linalg.inv( ((data_train.T) @ (data_train)) ) @ data_train.T @ target_train  

    # prediction
    y_pred = data_test @ weights

    # root mean squared error
    rmse = sklearn.metrics.mean_squared_error(target_test, y_pred, squared=False)
   
    return rmse


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    rmse = main(args)
    print("{:.2f}".format(rmse))
