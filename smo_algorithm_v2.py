#!/usr/bin/env python3
#
# Team members' IDs:
# 572e4c43-e678-11e9-9ce9-00505601122b  (Jaro Luknis)
# fa2094b3-2eab-11ec-986f-f39926f24a9c  (Jan Zubáč)
# a6ef6adf-e5c9-11e9-9ce9-00505601122b  (Ondřej Varga)
#
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from numpy.linalg import norm 


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--C", default=1, type=float, help="Inverse regularization strength")
parser.add_argument("--data_size", default=200, type=int, help="Data size")
parser.add_argument("--kernel", default="poly", type=str, help="Kernel type [poly|rbf]")
parser.add_argument("--kernel_degree", default=1, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--max_iterations", default=1000, type=int, help="Maximum number of iterations to perform")
parser.add_argument("--max_passes_without_as_changing", default=10, type=int, help="Number of passes without changes to stop after")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--tolerance", default=1e-7, type=float, help="Default tolerance for KKT conditions")
# If you add more arguments, ReCodEx will keep them with your default values.

def kernel_poly(x, y, gamma, kernel_degree ):
    return (gamma* (x.T @ y) + 1 ) **kernel_degree

def kernel_poly_matrix(data1, data2, gamma, kernel_degree ):
    # non-homog. kernel for data with 1 feature
    K = np.zeros((data1.shape[0], data2.shape[0]))
    for i, x_i in enumerate(data1):
        for j, x_j in enumerate(data2):
            K[i, j] = kernel_poly(x_i, x_j, gamma, kernel_degree )

    return K

def kernel_rbf(x, y, gamma ):
    return np.exp( - gamma* norm(x - y)**2 ) 

def kernel_rbf_matrix(data1, data2, gamma ):
    # rbf kernel for data with 1 feature
    K = np.zeros((data1.shape[0], data2.shape[0]))
    for i, x_i in enumerate(data1):
        for j, x_j in enumerate(data2):
            K[i, j] = kernel_rbf(x_i, x_j, gamma )

    return K


def predict(args: argparse.Namespace, x, y, t, a, b, K):

    y_pred = []

    for i in range(K.shape[0]):
        y_pred.append( ( a*t ) @ K[i].T + b)
    return y_pred

def kernel(args: argparse.Namespace, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # TODO: As in `kernel_linear_regression`, We consider the following `args.kernel`s:
    # - "poly": K(x, y; degree, gamma) = (gamma * x^T y + 1) ^ degree
    # - "rbf": K(x, y; gamma) = exp^{- gamma * ||x - y||^2}
    # raise NotImplementedError()
    K = np.zeros((x.shape[0], y.shape[0]))
    
    if args.kernel == 'poly':
        # K = np.outer(x, y)
        # K = np.outer(x, y)      # CHECK!!!
        # K = (args.kernel_gamma * K +1) ** args.kernel_degree  
        K = kernel_poly_matrix(x, y, args.kernel_gamma, args.kernel_degree)

    if args.kernel == 'rbf':
        # data_x_matrix = np.tile(x, (y.shape[0], 1)).T
        # data_y_matrix = np.tile(y, (x.shape[0], 1))
        # K = np.exp(-args.gamma  * (data_x_matrix - data_y_matrix)**2 )
        K = kernel_rbf_matrix(x, y, args.kernel_gamma )            # UGLY BUT CORRECT


    return K



# We implement the SMO algorithm as a separate method, so we can use
# it in the svm_multiclass assignment too.
def smo(
    args: argparse.Namespace,
    train_data: np.ndarray, train_target: np.ndarray,
    test_data: np.ndarray, test_target: np.ndarray
): # -> tuple[np.ndarray, np.ndarray, float, list[float], list[float]]:
    # Create initial weights
    a, b = np.zeros(len(train_data)), 0
    generator = np.random.RandomState(args.seed)

    # kernels
    K_train = kernel(args, train_data, train_data)
    K_test  = kernel(args, test_data, train_data)

    passes_without_as_changing = 0
    train_accs, test_accs = [], []
    for _ in range(args.max_iterations):
        as_changed = 0
        # Iterate through the data
        for i, j in enumerate(generator.randint(len(a) - 1, size=len(a))):
            # We want j != i, so we "skip" over the value of i
            j = j + (j >= i)
            
            # predictions 
            y_train =  (a * train_target) @ K_train[i].T + b    ### ???

            # errors
            Ei = y_train - train_target[i]
                                                                                                 # print("{}".format(_))
            # checking if NOT (kkt conditions)
            if (   (  (a[i] < (args.C - args.tolerance) ) and ( train_target[i]* Ei < - args.tolerance) ) or  ##  !!! Chyba
                   (  (a[i] >           args.tolerance )  and ( train_target[i]* Ei > args.tolerance) )   ):

                # calculation of 2nd derivative of L
                der2L = 2* K_train[i, j] - K_train[i,i] - K_train[j,j] 
                
                if der2L > - args.tolerance:                    # is not max -> next i
                    continue

                # calculation of a[j] unclipped using 2nd der
                y_train_j = (a * train_target) @ K_train[j].T + b    ### ???
                
                Ej = y_train_j - train_target[j]
                a_j_new = a[j] - ( ( train_target[j] * (Ei - Ej) ) /  der2L  )  

                # clipping of a_j_new to suitable 
                if ( train_target[i]  == train_target[j] ):    ## 
                    L = max(0, a[i] + a[j] - args.C)
                    H = min( args.C, a[i] + a[j] )
                else:
                    L = max(0, a[j] - a[i])
                    H = min(args.C, args.C + a[j] - a[i])
                a_j_new_clipped = max(L, a_j_new)           # clipping from lo side
                a_j_new_clipped = min(H, a_j_new_clipped)   # clipping from hi side

                if abs(a_j_new_clipped - a[j]) < args.tolerance:
                    continue   
                else:
                    # calculation of ai and b updates 
                    a_i_new = a[i] - train_target[i]*train_target[j] * ( a_j_new_clipped - a[j] )  ## ???

                    b_j_new = b - Ej  - train_target[i] * (a_i_new - a[i]) * K_train[i,j]
                    b_j_new = b_j_new - train_target[j] * (a_j_new_clipped - a[j]) * K_train[j,j]
                    
                    b_i_new = b - Ei  - train_target[i] * (a_i_new - a[i]) * K_train[i,i]
                    b_i_new = b_i_new - train_target[j] * (a_j_new_clipped - a[j]) * K_train[j,i]

                    if ( (a_i_new > args.tolerance ) and ( a_i_new < ( args.C - args.tolerance) ) ):   ### !!!! 
                        b_new = b_i_new
                    elif ( (a_j_new_clipped > args.tolerance ) and 
                         ( a_j_new_clipped < ( args.C - args.tolerance) ) ):
                        b_new = b_j_new
                    else:
                        b_new = ( b_i_new + b_j_new ) / 2

                    a[i], a[j], b = a_i_new, a_j_new_clipped, b_new    # updating all   
                    as_changed += 1

        # prediction
        # y_train, y_test = [], []
        # for i in range(len(train_data)):
        #     y_train_i = np.sum( (a * train_target) * K_train[i,:] ) + b
        #     y_train.append(y_train_i)

        # for i in range(len(test_data)):
        #     y_test_i = np.sum( (a * train_target) * K_test[i,:] ) + b   ### !! train_target
        #     y_test.append(y_test_i)

        y_train = predict(args, train_data, train_data, train_target, a, b, K_train)
        y_test  = predict(args, test_data,  train_data, train_target, a, b, K_test)


        y_train, y_test = np.sign(y_train), np.sign(y_test) 

        train_acc = sklearn.metrics.accuracy_score(y_train, train_target )
        test_acc  = sklearn.metrics.accuracy_score(y_test, test_target)

        # TODO: After each iteration, measure the accuracy for both the
        # train set and the test set and append it to `train_accs` and `test_accs`.
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        # Stop training if max_passes_without_as_changing passes were reached
        passes_without_as_changing = 0 if as_changed else passes_without_as_changing + 1
        if passes_without_as_changing >= args.max_passes_without_as_changing:
            break

        if len(train_accs) % 100 == 0 and len(train_accs) < args.max_iterations:
            print("Iteration {}, train acc {:.1f}%, test acc {:.1f}%".format(
                len(train_accs), 100 * train_accs[-1], 100 * test_accs[-1]))

    # TODO: Create an array of support vectors (in the same order in which they appeared
    # in the training data; to avoid rounding errors, consider a training example
    # a support vector only if a_i > `args.tolerance`) and their weights (a_i * t_i).
    support_vectors, support_vector_weights = [], []
    for i in range(a.shape[0]):
        if a[i] > args.tolerance:
            support_vectors.       append(train_data[i])
            support_vector_weights.append(a[i]*train_target[i])


    print("Done, iteration {}, support vectors {}, train acc {:.1f}%, test acc {:.1f}%".format(
        len(train_accs), len(support_vectors), 100 * train_accs[-1], 100 * test_accs[-1]))

    return support_vectors, support_vector_weights, b, train_accs, test_accs

def main(args: argparse.Namespace): # -> tuple[np.ndarray, np.ndarray, float, list[float], list[float]]:
    # Generate an artifical regression dataset, with +-1 as targets
    data, target = sklearn.datasets.make_classification(
        n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)
    target = 2 * target - 1

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Run the SMO algorithm
    support_vectors, support_vector_weights, bias, train_accs, test_accs = smo(
        args, train_data, train_target, test_data, test_target)

    if args.plot:
        import matplotlib.pyplot as plt
        def plot(predict, support_vectors):
            xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
            ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
            predictions = [[predict(np.array([x, y])) for x in xs] for y in ys]
            test_mismatch = np.sign([predict(x) for x in test_data]) != test_target
            plt.figure()
            plt.contourf(xs, ys, predictions, levels=0, cmap=plt.cm.RdBu)
            plt.contour(xs, ys, predictions, levels=[-1, 0, 1], colors="k", zorder=1)
            plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, marker="o", label="Train", cmap=plt.cm.RdBu, zorder=2)
            plt.scatter(support_vectors[:, 0], support_vectors[:, 1], marker="o", s=90, label="Support Vectors", c="#00dd00")
            plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target, marker="*", label="Test", cmap=plt.cm.RdBu, zorder=2)
            plt.scatter(test_data[test_mismatch, 0], test_data[test_mismatch, 1], marker="*", s=130, label="Test Errors", c="#ffff00")
            plt.legend(loc="upper center", ncol=4)

        # If you want plotting to work (not required for ReCodEx), you need to
        # define `predict_function` computing SVM value `y(x)` for the given x.
        predict_function = lambda x: None

        plot(predict_function, support_vectors)
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return support_vectors, support_vector_weights, bias, train_accs, test_accs

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
