#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection



parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--epochs", default=50, type=int, help="Number of SGD iterations over the data")
parser.add_argument("--l2", default=0.0, type=float, help="L2 regularization strength")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace): # -> tuple[float, float]:
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Generate an artifical regression dataset
    data, target = sklearn.datasets.make_regression(n_samples=args.data_size, random_state=args.seed)

    # adding bias as an extra feature (column of ones)
    ones = np.ones((data.shape[0], 1))
    data = np.concatenate((data,ones),axis = 1)

    # train/test splitting of the dataset
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(data, target, 
    test_size=args.test_size, random_state=args.seed)

    # generate initial linear regression weights
    weights = generator.uniform(size=train_data.shape[1])

    train_rmses, test_rmses = [], []
    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])  # generation of random permutation
        grads = np.zeros(( len(weights) , args.batch_size))       # prealocation of grads
        
        for i in range(0, len(permutation), args.batch_size): 
            perm_batch = permutation[i:(i+args.batch_size)]
            xb_T = train_data[perm_batch,:]    # batch data
            tb = train_target[perm_batch]      # batch target

            # gradient calculation
            for j in range(args.batch_size):
                grads[:,j] = ((xb_T[j,:] @ weights) - tb[j] ) * xb_T[j,:]    # grad for each point: (x_j^T weights - t_j) * x_j
            gradient = np.mean(grads, axis=1)  # avg gradient

            # update of weights
            weights = weights - args.learning_rate * (gradient + args.l2 * weights)
         

        # prediction and rmse calculation
        y_test, y_train = test_data @ weights, train_data @ weights 
        train_rmse = np.sqrt(  np.sum( (y_train - train_target)**2 )/len(y_train)  ) 
        test_rmse  = np.sqrt(  np.sum( (y_test  - test_target)**2 )/len(y_test)  )
        train_rmses.append(train_rmse)
        test_rmses.append(test_rmse)
                 

    # TODO: Compute into `explicit_rmse` test data RMSE when fitting
    # `sklearn.linear_model.LinearRegression` on train_data (ignoring args.l2)
    model = sklearn.linear_model.LinearRegression()
    model.fit(train_data, train_target)
    y_pred = model.predict(test_data)
    
    explicit_rmse = np.sqrt( np.sum( (y_pred - test_target)**2 )/len(y_pred) )
    # explicit_rmse = np.sqrt( sklearn.metrics.mean_squared_error(train_target, y_train) )
    # print(explicit_rmse)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(train_rmses, label="Train")
        plt.plot(test_rmses, label="Test")
        plt.xlabel("Iterations")
        plt.ylabel("RMSE")
        plt.legend()
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return test_rmses[-1], explicit_rmse

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    sgd_rmse, explicit_rmse = main(args)
    print("Test RMSE: SGD {:.2f}, explicit {:.2f}".format(sgd_rmse, explicit_rmse))

# moje
# main(args)