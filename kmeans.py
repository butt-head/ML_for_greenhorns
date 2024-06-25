#!/usr/bin/env python3
import argparse

import numpy as np

import sklearn.datasets

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--clusters", default=3, type=int, help="Number of clusters")
parser.add_argument("--examples", default=200, type=int, help="Number of examples")
parser.add_argument("--init", default="random", type=str, help="Initialization (random/kmeans++)")
parser.add_argument("--iterations", default=20, type=int, help="Number of kmeans iterations to perfom")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
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


def plot(args: argparse.Namespace, iteration: int,
         data: np.ndarray, centers: np.ndarray, clusters: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    if args.plot is not True:
        if not plt.gcf().get_axes(): plt.figure(figsize=(4*2, 5*6))
        plt.subplot(6, 2, 1 + len(plt.gcf().get_axes()))
    plt.title("KMeans Initialization" if not iteration else
              "KMeans After Iteration {}".format(iteration))
    plt.gca().set_aspect(1)
    plt.scatter(data[:, 0], data[:, 1], c=clusters)
    plt.scatter(centers[:, 0], centers[:, 1], marker="P", s=200, c="#ff0000")
    plt.scatter(centers[:, 0], centers[:, 1], marker="P", s=50, c=range(args.clusters))
    if args.plot is True: plt.show()
    else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

def main(args: argparse.Namespace) -> np.ndarray:
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Generate artificial data
    data, target = sklearn.datasets.make_blobs(
        n_samples=args.examples, centers=args.clusters, n_features=2, random_state=args.seed)

    # TODO: Initialize `centers` to be
    # - if args.init == "random", K random data points, using the indices
    #   returned by
    #     generator.choice(len(data), size=args.clusters, replace=False)
    # - if args.init == "kmeans++", generate the first cluster by
    #     generator.randint(len(data))
    #   and then iteratively sample the rest of the clusters proportionally to
    #   the square of their distances to their closest cluster using
    #     generator.choice(unused_points_indices, p=square_distances / np.sum(square_distances))
    #   Use the `np.linalg.norm` to measure the distances.
    # 
     
    if args.init == "random":
        inds = generator.choice(len(data), size=args.clusters, replace=False)
        centers = data[inds]

    if args.init == "kmeans++":
        centers = np.zeros((args.clusters, data.shape[1]))
        ind = generator.randint(len(data))
        
        centers[0] = data[ind]
        ind_unused = [ i for i in range(len(data)) if i!=ind]

        inds = [ind]
        for k in range(1,args.clusters):
            distances = distance_matrix(data[ind_unused], centers[:k])
            distances_to_nearest = np.min(distances, axis=-1) 
            ind = generator.choice(ind_unused, p=distances_to_nearest**2 / np.sum(distances_to_nearest**2))
            inds.append(ind)
            centers[k] = data[ind]

            ind_unused = [ i for i in range(len(data)) if i not in inds]
        

    if args.plot:
        plot(args, 0, data, centers, clusters=None)

    # Run `args.iterations` of the K-Means algorithm.
    for iteration in range(args.iterations):
        # TODO: Perform a single iteration of the K-Means algorithm, storing
        #   to `clusters`.

        z = np.zeros((data.shape[0], args.clusters))
        distances = distance_matrix(data, centers)
        for i, ind_min in enumerate(np.argmin(distances, axis=-1)): 
            z[i,ind_min] = 1


        for k in range(args.clusters):
            centers[k] = np.sum(z[:,k][:,None] * data, axis=0)/np.sum(z[:,k])

        clusters = np.argmax(z, axis=-1)
        if args.plot:
            plot(args, 1 + iteration, data, centers, clusters)

    return clusters

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    centers = main(args)
    print("Cluster assignments:", centers, sep="\n")
