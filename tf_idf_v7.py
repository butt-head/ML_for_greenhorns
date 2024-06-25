#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import sys
import urllib.request

import numpy as np
import sklearn.metrics
import sklearn.model_selection
import sklearn.neighbors

# my imports
from pprint import pprint
import re
import time


class NewsGroups:
    def __init__(self,
                 name="20newsgroups.train.pickle",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        with lzma.open(name, "rb") as dataset_file:
            dataset = pickle.load(dataset_file)

        self.DESCR = dataset.DESCR
        self.data = dataset.data[:data_size]
        self.target = dataset.target[:data_size]
        self.target_names = dataset.target_names

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--idf", default=False, action="store_true", help="Use IDF weights")
parser.add_argument("--k", default=1, type=int, help="K nearest neighbors to consider")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=37, type=int, help="Random seed")
parser.add_argument("--tf", default=False, action="store_true", help="Use TF weights")
parser.add_argument("--test_size", default=1000, type=int, help="Test set size")
parser.add_argument("--train_size", default=1000, type=int, help="Train set size")
# For these and any other arguments you add, ReCodEx will keep your default value.


def create_features(data, min_word_count = 2, min_len = 2):
    "creates a feature for every word that is present at least twice in the training data."

    data_str = " ".join(data)
    data_words =  re.findall( r'(\w+)',  data_str)
    data_unique_words, w_counts    = np.unique(data_words, return_counts=True)
    
    features = []
    for w, wc in zip(data_unique_words, w_counts):
        if wc >= min_word_count and len(w) >=min_len:
            features.append(w)
    features = dict( zip(features, range(len(features))) )
    return features


def extract_features_from_document(document, features, min_len = 2):

    data_document_str = "".join(document)
    document_words =  re.findall( r'(\w+)',  data_document_str)
    document_unique_words, w_counts_docu = np.unique(document_words, return_counts=True)

    word_features_extract, word_features_count = [], []
    for uw, wc in zip(document_unique_words, w_counts_docu):
        # if (len(uw) >=min_len) and (uw in features):      ## DRUHA PODMINKA TRVA DLOUHO
        if (len(uw) >=min_len):
            if (uw in features.keys()):
                word_features_extract.append(uw),  word_features_count.append(wc)

    return dict( zip(word_features_extract, word_features_count) )


def transform_data_tf_idf(args, data, features, calculate_idf=True, idf = None, normalize=True):

    dataset = np.zeros( (len(data), len(features)) )
    n_docs_with_term = np.ones(len(features))     # + 1 for calculation of idf
    
    for i, d in enumerate(data):

        features_docu = extract_features_from_document(d, features)  # extraction from one document
        tfs = np.array(list(features_docu.values()))/len(features_docu)

        intersect = features.keys() & features_docu.keys() 
        js = np.sort([ features[i] for i in intersect ])
        n_docs_with_term[js] += 1
       
        if args.tf:
            dataset[i,js] = tfs
        else:
            dataset[i,js] = 1

    if args.idf:
        if calculate_idf:
            idf = np.log(len(data)/ n_docs_with_term)
        else:
            idf = idf
        # dataset = dataset @ idf                            ### !!
        dataset = dataset * idf

    if normalize:
        dataset_norm =  np.tile( np.linalg.norm(dataset, axis = -1), (len(features),1) ).T
        # dataset_norm =  np.linalg.norm(dataset, axis = -1)
        dataset = (1/dataset_norm) * dataset  

    return dataset, tfs, idf


def main(args: argparse.Namespace) -> float:
    # Load the 20newsgroups data.
    newsgroups = NewsGroups(data_size=args.train_size + args.test_size)

    # Create train-test split.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        newsgroups.data, newsgroups.target, test_size=args.test_size, random_state=args.seed)
    
    time0 = time.time()

    # creating features over all documents in training set
    features = create_features(train_data)
    print(len(features))
    
    dataset_train, tf_train, idf_train = transform_data_tf_idf(args, train_data, features, calculate_idf=True, normalize=True)
    dataset_test, tf_test, idf_test = transform_data_tf_idf(args, test_data, features, calculate_idf=False, idf=idf_train, normalize=True)

    model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=args.k, algorithm="brute", p=2)   # p=2 ... Minkowski->Euclidean metric
    model.fit(dataset_train, train_target)
    y_test_predict = model.predict(dataset_test) 


    # TODO: Create a feature for every word that is present at least twice
    # in the training data. A word is every maximal sequence of at least 2 word characters,
    # where a word character corresponds to a regular expression `\w`.

    # TODO: For each document, compute its features as
    # - term frequency(TF), if `args.tf` is set;
    # - otherwise, use binary indicators (1 if a given term is present, else 0)
    #
    # Then, if `args.idf` is set, multiply the document features by the
    # inverse document frequencies (IDF), where
    # - use the variant which contains `+1` in the denominator;
    # - the IDFs are computed on the train set and then reused without
    #   modification on the test set.

    # TODO: Perform classification of the test set using the k-NN algorithm
    # from sklearn (pass the `algorithm="brute"` option), with `args.k` nearest
    # neighbors. For TF-IDF vectors, the cosine similarity is usually used, where
    #   cosine_similarity(x, y) = x^T y / (||x|| * ||y||).
    #
    # To employ this metric, you have several options:
    # - you could try finding out whether `KNeighborsClassifier` supports it directly;
    # - or you could compute it yourself, but if you do, you have to precompute it
    #   in a vectorized way, so using `metric="precomputed"` is fine, but passing
    #   a callable as the `metric` argument is not (it is too slow);
    # - finally, the nearest neighbors according to cosine_similarity are equivalent to
    #   the neighbors obtained by the usual Euclidean distance on L2-normalized vectors.

    # TODO: Evaluate the performance using macro-averaged F1 score.
    # f1_score = None

    f1_score = sklearn.metrics.f1_score(test_target, y_test_predict, average="macro")


    # print("Total duration: {}s\n".format(time.time()-time0))


    return f1_score

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    #main(args)
    f1_score = main(args)
    # print(100 * f1_score)
    print("F-1 score for TF={}, IDF={}, k={}: {:.1f}%".format(args.tf, args.idf, args.k, 100 * f1_score))
