#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection


import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=5, type=int, help="Batch size")
parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.




# def softmax(z, subtract_max_z = True):
#     if subtract_max_z:
#         z = z - np.max(z)
#     return np.exp(z)/np.sum(np.exp(z))


def softmax(z, subtract_max_z = True):
    if subtract_max_z:
        z = z - np.max(z, axis=-1, keepdims=True)

    return np.exp(z)/np.sum(np.exp(z), axis=-1, keepdims=True)


def one_hot_matrix(tb,k=10):

    one_hot_matrix = np.eye(k)
    ind_tb = [int(t) for t in tb]

    return one_hot_matrix[ind_tb,:]
    # return one_hot_matrix[:,ind_tb]


def log_loss(y_pred, target):
    
    n_classes = y_pred.shape[-1]
    H_pq = n_classes*np.mean(-one_hot_matrix(target)*np.log(y_pred))

    return H_pq

def accuracy(y_pred, target):

    H_pq = np.mean( target == np.argmax(y_pred, axis=-1) )
    
    return H_pq


def main(args: argparse.Namespace): # -> tuple[np.ndarray, list[tuple[float, float]]]:
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Use the digits dataset
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Append a constant feature with value 1 to the end of every input data
    data = np.pad(data, ((0, 0), (0, 1)), constant_values=1)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    #print(train_target)

    # Generate initial model weights
    weights = generator.uniform(size=[train_data.shape[1], args.classes], low=-0.1, high=0.1)  # D x K 
    # print('weights.shape', weights.shape)  # D X K

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])
        # print('train_data.shape', train_data.shape)  # 1000 X D

        for i in range(0, len(permutation), args.batch_size): 
            perm_batch = permutation[i:(i+args.batch_size)]
            xb_T = train_data[perm_batch,:]    # batch data
            tb = train_target[perm_batch]      # batch target
            
            
            xb = np.transpose(xb_T)
            gradient = xb @ (softmax(xb_T @ weights) - one_hot_matrix(tb))
            gradient = gradient/len(perm_batch)
            
            # update of weights
            weights = weights - args.learning_rate * gradient
       
        y_test, y_train = softmax(test_data @ weights), softmax(train_data @ weights)


        
        # plt.plot(range(100), test_target[:100], 'o-', label='test target')
        # plt.plot(range(100), ymax[:100], 'o-', label='test pred')
        # plt.legend()
        # plt.show()

        # print(y_train)
        

        # TODO: After the SGD epoch, measure the average loss and accuracy for both the
        # train test and the test set. The loss is the average MLE loss (i.e., the
        # negative log likelihood, or crossentropy loss, or KL loss) per example.
        
        train_accuracy, train_loss, test_accuracy, test_loss = None, None, None, None


        # test_loss = sklearn.metrics.log_loss(test_target, y_test, normalize=True)
        # train_loss = sklearn.metrics.log_loss(train_target, y_train, normalize=True)
        
        # print("train_loss:", train_loss)
        # print("test_loss:", test_loss)



        train_loss = log_loss(y_train, train_target)  
        test_loss =  log_loss(y_test,  test_target)
        train_accuracy = accuracy(y_train, train_target)
        test_accuracy = accuracy(y_test,  test_target)
        
        # print("train_loss my:", train_loss)
        # print("test_loss my:", test_loss)

        # print("test_target:",test_target)
        # print("y_test:",y_test)


        print("After epoch {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
            epoch + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

    return weights, [(train_loss, train_accuracy), (test_loss, test_accuracy)]

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(args)
    print("Learned weights:", *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")
