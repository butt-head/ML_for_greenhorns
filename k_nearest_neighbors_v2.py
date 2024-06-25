#!/usr/bin/env python3
import argparse
import os
import sys
import urllib.request

import numpy as np
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing

import time

class MNIST:
    """MNIST Dataset.

    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in range 0-255, the targets are numbers 0-9.
    """
    def __init__(self,
                 name="mnist.train.npz",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset, i.e., `data` and optionally `target`.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value[:data_size])
        self.data = self.data.reshape([-1, 28*28]).astype(np.float)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--k", default=3, type=int, help="K nearest neighbors to consider")
parser.add_argument("--p", default=2, type=int, help="Use L_p as distance metric")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=1000, type=int, help="Test set size")
parser.add_argument("--train_size", default=1000, type=int, help="Train set size")
# parser.add_argument("--weights", default="uniform", type=str, help="Weighting to use (uniform/inverse/softmax)")
parser.add_argument("--weights", default="inverse", type=str, help="Weighting to use (uniform/inverse/softmax)")

# If you add more arguments, ReCodEx will keep them with your default values.


def norm(x, p=2):
    norm = ( np.sum( np.abs(x)**p , axis=-1, keepdims=True) )**(1/p)
    return norm

def distance(x1, x2, p=2):
    # assert(x1.shape == x2.shape)
    return norm(x2-x1, p=p)


def distance_line(vec, data2, p=2):
    dist_vec_data = np.array([item for sublist in distance(vec, data2, p=p) for item in sublist])
    return dist_vec_data

def distance_matrix(data1, data2, p=2):
    distance_mat = np.full( (data1.shape[0], data2.shape[0]), np.inf )
    for j in range(data1.shape[0]):
        distance_mat[j,:] = distance_line(data1[j,:], data2, p=p)

    return distance_mat

def softmax(z, subtract_max_z = True):
    if subtract_max_z:
        z = z - np.max(z, axis=-1, keepdims=True)

    return np.exp(z)/np.sum(np.exp(z), axis=-1, keepdims=True)



def main(args: argparse.Namespace): #-> float:
    # Load MNIST data, scale it to [0, 1] and split it to train and test.
    mnist = MNIST(data_size=args.train_size + args.test_size)
    mnist.data = sklearn.preprocessing.MinMaxScaler().fit_transform(mnist.data)
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        mnist.data, mnist.target, test_size=args.test_size, random_state=args.seed)


    # print("train_data.shape:", train_data.shape)

    # TODO: Generate `test_predictions` with classes predicted for `test_data`.
    #
    # Find `args.k` nearest neighbors, choosing the ones with the smallest train_data
    # indices in case of ties. Use the most frequent class (optionally weighted
    # by a given scheme described below) as prediction, choosing the one with the
    # smallest class index when there are multiple classes with the same frequency.
    #
    # Use L_p norm for a given p (either 1, 2 or 3) to measure distances.
    #
    # The weighting can be:
    # - "uniform": all nearest neighbors have the same weight
    # - "inverse": `1/distances` is used as weights
    # - "softmax": `softmax(-distances)` is used as weights
    #
    # If you want to plot misclassified examples, you also need to fill `test_neighbors`
    # with indices of nearest neighbors; but it is not needed for passing in ReCodEx.

    t0 = time.time()
    distances_test_train = distance_matrix(test_data, train_data, p=args.p)
    kNN_tt_inds = np.argpartition(distances_test_train, kth=args.k)[:, 0:args.k]
    t = time.time() -t0

    t0 = time.time()
    test_predictions = np.zeros(test_target.shape)
    test_neighbors = []
    for i, test_t in enumerate(test_target):
        
        test_neighbors.append(kNN_tt_inds[i,:])

        if args.weights == "uniform":
            counts = np.bincount(train_target[kNN_tt_inds[i,:]])
            test_predictions[i] = np.argmax(counts)          # print(np.argmax(counts))
        else:
            classes = np.unique(train_target[kNN_tt_inds[i,:]])
            weights_k = np.zeros(len(classes))

            if args.weights == "inverse":
                weights = 1/distances_test_train[i, kNN_tt_inds[i,:]]    # weights - reciprocal distances

            if args.weights == "softmax":
                weights = softmax(-distances_test_train[i, kNN_tt_inds[i,:]])
                
            for w, target in zip(weights, train_target[kNN_tt_inds[i,:]]):
                for j, k in enumerate(classes):
                    if target==k:
                        weights_k[j] += w
            test_predictions[i] = classes[np.argmax(weights_k)]
        t1 = time.time() -t0      

    accuracy = sklearn.metrics.accuracy_score(test_target, test_predictions)

    if args.plot:
        import matplotlib.pyplot as plt
        examples = [[] for _ in range(10)]
        for i in range(len(test_predictions)):
            if test_predictions[i] != test_target[i] and not examples[test_target[i]]:
                examples[test_target[i]] = [test_data[i], *train_data[test_neighbors[i]]]
        examples = [[img.reshape(28, 28) for img in example] for example in examples if example]
        examples = [[example[0]] + [np.zeros_like(example[0])] + example[1:] for example in examples]
        plt.imshow(np.concatenate([np.concatenate(example, axis=1) for example in examples], axis=0), cmap="gray")
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")
 
    return accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    accuracy = main(args)
    print("K-nn accuracy for {} nearest neighbors, L_{} metric, {} weights: {:.2f}%".format(
        args.k, args.p, args.weights, 100 * accuracy))
