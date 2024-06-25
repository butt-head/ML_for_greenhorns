#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.metrics
from numpy.linalg import norm 
import copy


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
parser.add_argument("--data_size", default=50, type=int, help="Data size")
parser.add_argument("--kernel", default="rbf", type=str, help="Kernel type [poly|rbf]")
parser.add_argument("--kernel_degree", default=3, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--iterations", default=200, type=int, help="Number of training iterations")
parser.add_argument("--l2", default=0.0, type=float, help="L2 regularization weight")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# If you add more arguments, ReCodEx will keep them with your default values.


def kernel_rbf(x, y, gamma ):
    return np.exp( - gamma* norm(x - y)**2 ) 

def kernel_poly_matrix(data1, data2, gamma, kernel_degree ):
    # non-homog. kernel for data with 1 feature
    K = np.outer(data1, data2)
    K = (gamma * K +1) ** kernel_degree  
    return K


def kernel_rbf_matrix(data1, data2, gamma ):
    # rbf kernel for data with 1 feature
    data1_matrix = np.tile(data1, (data2.shape[0], 1)).T
    data2_matrix = np.tile(data2, (data1.shape[0], 1))

    K = np.exp(-gamma * (data1_matrix - data2_matrix)**2 )
    return K


def main(args: argparse.Namespace): # -> tuple[list[float], list[float]]:
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Generate an artifical regression dataset
    train_data = np.linspace(-1, 1, args.data_size)
    train_target = np.sin(5 * train_data) + generator.normal(scale=0.25, size=args.data_size) + 1

    test_data = np.linspace(-1.2, 1.2, 2 * args.data_size)
    test_target = np.sin(5 * test_data) + 1

    betas = np.zeros(args.data_size)
    # TODO: Perform `args.iterations` of SGD-like updates, but in dual formulation
    # using `betas` as weights of individual training examples.
    #
    # We assume the primary formulation of our model is
    #   y = phi(x)^T w + bias
    # and the loss in the primary problem is batched MSE with L2 regularization:
    #   L = sum_{i \in B} 1/|B| * [1/2 * (phi(x_i)^T w + bias - target_i)^2] + 1/2 * args.l2 * w^2
    # Regarding the L2 regularization, note that it always affects all betas, not   !!!
    # just the ones in the batch.
    #
    # DONE For `bias`, explicitly use the average of the training targets, and do
    # not update it further during training.
    #
    # Instead of using feature map `phi` directly, we use a given kernel computing
    # DONE  K(x, y) = phi(x)^T phi(y)
    # We consider the following `args.kernel`s:
    # DONE - "poly": K(x, y; degree, gamma) = (gamma * x^T y + 1) ^ degree
    # DONE - "rbf": K(x, y; gamma) = exp^{- gamma * ||x - y||^2}
    #
    # After each iteration, compute RMSE both on training and testing data.
    train_rmses, test_rmses = [], []

    bias_train = np.mean(train_target)
    bias_test = np.mean(test_target)
    
    if args.kernel == "poly":
        K_train = kernel_poly_matrix(train_data, train_data, args.kernel_gamma, args.kernel_degree)
        K_test  = kernel_poly_matrix(test_data, train_data, args.kernel_gamma, args.kernel_degree)

    if args.kernel == "rbf":
        K_train = kernel_rbf_matrix(train_data, train_data, args.kernel_gamma)
        K_test = kernel_rbf_matrix(test_data, train_data, args.kernel_gamma)

    print("K_train.shape",K_train.shape)
    print("K_test.shape",K_test.shape)

    for iteration in range(args.iterations):
        permutation = generator.permutation(train_data.shape[0])

        
        for i in range(0, len(permutation), args.batch_size):
            batch = permutation[i:(i+args.batch_size)]
            betas_new = np.zeros(betas.shape)
            betas_new[batch] =  K_train[batch,:] @ betas - train_target[batch] + bias_train

            betas = betas + -(args.learning_rate/args.batch_size) * betas_new - (args.learning_rate* args.l2) * betas

        y_train = K_train @ betas + bias_train
        y_test  = K_test  @ betas + bias_test 

        train_rmse = np.sqrt( np.sum( (y_train - train_target)**2 )/len(y_train) )
        test_rmse = np.sqrt( np.sum( (y_test - test_target)**2 )/len(y_test) )
        train_rmses.append(train_rmse) 
        test_rmses.append(test_rmse)


        # TODO: Process the data in the order of `permutation`, performing
        # batched updates to the `betas`. You can assume that `args.batch_size`
        # exactly divides `train_data.shape[0]`.

        # TODO: Append RMSE on training and testing data to `train_rmses` and
        # `test_rmses` after the iteration.

        # if (iteration + 1) % 1 == 0:
        #     print("Iteration {}, train RMSE {:.2f}".format(
        #         iteration + 1, train_rmses[-1]))

        if (iteration + 1) % 10 == 0:
            print("Iteration {}, train RMSE {:.2f}, test RMSE {:.2f}".format(
                iteration + 1, train_rmses[-1], test_rmses[-1]))

    if args.plot:
        import matplotlib.pyplot as plt
        # If you want the plotting to work (not required for ReCodEx), compute the `test_predictions`.
        test_predictions = y_test

        plt.plot(train_data, train_target, "bo", label="Train target")
        plt.plot(test_data, test_target, "ro", label="Test target")
        plt.plot(test_data, test_predictions, "g-", label="Predictions")
        plt.legend()
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return train_rmses, test_rmses

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
