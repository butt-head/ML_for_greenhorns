#!/usr/bin/env python3
#
# Team members' IDs:
# 572e4c43-e678-11e9-9ce9-00505601122b  (Jaro Luknis)
# fa2094b3-2eab-11ec-986f-f39926f24a9c  (Jan Zubáč)
# a6ef6adf-e5c9-11e9-9ce9-00505601122b  (Ondřej Varga)
#
#
import argparse

import numpy as np
import scipy.stats

import sklearn.datasets
import sklearn.model_selection


from pprint import pprint


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Smoothing parameter for Bernoulli and Multinomial NB")
parser.add_argument("--naive_bayes_type", default="gaussian", type=str, help="NB type to use")
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.


def data_to_classes(data, target, args):

    data_classes = []
    for k in range(args.classes):
        data_classes.append(data[np.where(k == target)])

    return np.array(data_classes, dtype=object)
        


def calculate_mean(data_classes):

    n_features = data_classes[0].shape[1]
    n_classes  = data_classes.shape[0]
    mean = np.zeros((n_features, n_classes))    # shape D x K (features X classes)
    
    for k in range(n_classes):
        mean[:,k] = np.mean(data_classes[k], axis = 0) 

    return mean

def calculate_variance(data_classes, mean, alpha):

    n_features = data_classes[0].shape[1]
    n_classes  = data_classes.shape[0]

    variance = np.zeros((n_features, n_classes))    # shape D x K  (features X classes)
    
    for k in range(n_classes):
        variance[:,k] = np.sum( (data_classes[k] - mean[:,k])**2 , axis = 0)/data_classes[k].shape[0]

    variance = variance + alpha   # variance smoothing

    return variance



def calculate_prior_classes(data_classes, n_classes):
    priors = np.zeros(n_classes) 

    for k in range(n_classes):
        priors[k] = len(data_classes[k]) 

    priors = priors/np.sum(priors)         # normalization

    return priors
 

def predict_gaussian(test_data, mean, variance, priors_classes):
    
    n_classes = mean.shape[1]

    gauss_log = np.zeros((n_classes, *test_data.shape))
    for k in range(n_classes):
        gauss_log[k, :, :] =  scipy.stats.norm.logpdf(test_data, mean[:,k], np.sqrt(variance[:,k]))   # calculation of probability density function for particular classes


    gauss_log = np.sum(gauss_log, axis=2)                                       # sum over the features
    predictions = np.argmax( gauss_log.T + np.log(priors_classes), axis=1 )     # argmax over classes

    return predictions



def calculate_probabilities(data_classes, alpha, binary_features = True):

    n_features = data_classes[0].shape[1]
    n_classes  = data_classes.shape[0]
    
    prob = np.zeros((n_features, n_classes))    # shape D x K (features X classes)

    # print("data_classes[0].shape", data_classes[0].shape)
    for k in range(n_classes):
        if binary_features:
            data_classes_bin_k = np.zeros(data_classes[k].shape) 
            data_classes_bin_k[data_classes[k]>=8] = 1
            N_k = data_classes_bin_k.shape[0]
            p_d_k = ( np.sum(data_classes_bin_k, axis=0) + alpha )/( N_k + 2*alpha)
        
        else:
            n_d_k = np.sum(data_classes[k], axis=0)
            p_d_k = (n_d_k + alpha)/ (np.sum(n_d_k) + alpha*n_features) 


        prob[:,k] = p_d_k

    return prob


def predict(test_data, probabilities, priors_classes, type = 'bernoulli'):
    
    n_classes = priors_classes.shape[0]
    y = np.zeros((n_classes, *test_data.shape))

    test_data_bin = np.zeros(test_data.shape)
    test_data_bin[test_data>=8] = 1 

    p = probabilities
    for k in range(n_classes):
        if type == 'bernoulli':
            y[k, :, :] = test_data_bin * np.log(p[:,k]/(1 - p[:,k])) +  np.log(1 - p[:,k])
        if type == 'multinomial':
             y[k, :, :] = test_data * np.log(p[:,k])     

    y = np.sum(y, axis=2)                                               # sum over the features
    predictions = np.argmax( y.T + np.log(priors_classes), axis=1 )     # argmax over classes


    return predictions


def main(args: argparse.Namespace) -> float:

    # TODO: Train a naive Bayes classifier on the train data.
    #
    # The `args.naive_bayes_type` can be one of:
    # - "gaussian": implement Gaussian NB training, by estimating mean and
    #   variance of the input features. For variance estimation use
    #     1/N * \sum_x (x - mean)^2
    #   and additionally increase all estimated variances by `args.alpha`.
    #
    #   During prediction, you can compute probability density function of a Gaussian
    #   distribution using `scipy.stats.norm`, which offers `pdf` and `logpdf`
    #   methods, among others.
    #
    # - "multinomial": Implement multinomial NB with smoothing factor `args.alpha`.
    #
    # - "bernoulli": Implement Bernoulli NB with smoothing factor `args.alpha`.
    #   Because Bernoulli NB works with binary data, binarize the features as
    #   [feature_value >= 8], i.e., consider a feature as one iff it is >= 8,
    #   during both estimation and prediction.
    #
    #
    #
    # features d  (1 to D)
    # classes k (1 to K)
    # gaussian bayes:
    # mu_d_k
    # sigma2_d_k
    #
    # naive bayes:
    #   p(Ck|x)                 = p(x|Ck)            * p(Ck)
    #   (probability of          (gaussian,                 (prior of class - class frequency)
    #   class, features         probability of feature,
    #   given)                  class given, 
    #                           "naive" part: 
    #                           features for given class independent)   



    # Use the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)


    # splitting the train data to classes 
    train_data_classes = data_to_classes(train_data, train_target, args)
    priors_classes = calculate_prior_classes(train_data_classes, args.classes)

    # Gaussian
    if args.naive_bayes_type == "gaussian":
        mean_train     = calculate_mean(train_data_classes)                                 # training phase - estimating mean and variance
        variance_train = calculate_variance(train_data_classes, mean_train, args.alpha)
        predictions = predict_gaussian(test_data, mean_train, variance_train, priors_classes)


    # Bernoulli
    if args.naive_bayes_type == "bernoulli":
        probabilities = calculate_probabilities(train_data_classes, args.alpha, binary_features=True)
        predictions = predict(test_data, probabilities, priors_classes, type = 'bernoulli')


    # Multinomial
    if args.naive_bayes_type == "multinomial":
        probabilities = calculate_probabilities(train_data_classes, args.alpha, binary_features=False)
        predictions = predict(test_data, probabilities, priors_classes, type = 'multinomial')


    # Predict the test data classes and compute test accuracy.
    test_accuracy = sklearn.metrics.accuracy_score(test_target, predictions)


    return test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(args)

    print("Test accuracy {:.2f}%".format(100 * test_accuracy))
