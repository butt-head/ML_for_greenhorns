#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=13, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace): #-> tuple[float, float]:
    # Linear regression with L2 regularization
    dataset = sklearn.datasets.load_diabetes()

    data, target = dataset.data, dataset.target
    data_train, data_test, target_train, target_test = sklearn.model_selection.train_test_split(data, target, test_size=args.test_size, random_state=args.seed)

    lambdas = np.geomspace(0.01, 10, num=500)

    rmses = []
    for l in lambdas:
        model = sklearn.linear_model.Ridge(alpha = l) 
        model.fit(data_train, target_train)
        y_pred = model.predict(data_test)              
        rmse = np.sqrt( np.sum( (y_pred - target_test)**2 )/len(y_pred) )
          
        rmses.append(rmse)

    
    best_rmse = np.min(rmses)
    best_lambda = lambdas[np.argmin(rmses)]


    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(lambdas, rmses)
        plt.xscale("log")
        plt.xlabel("L2 regularization strength")
        plt.ylabel("RMSE")
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return best_lambda, best_rmse


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    best_lambda, best_rmse = main(args)
    print("{:.2f} {:.2f}".format(best_lambda, best_rmse))
