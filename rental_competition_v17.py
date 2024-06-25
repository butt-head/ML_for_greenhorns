#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import urllib.request
import numpy as np

# my imports
import sklearn.model_selection
import sklearn.pipeline
import sklearn.compose
import sklearn.preprocessing
import sklearn.linear_model


class Dataset:
    """Rental Dataset.

    The dataset instances consist of the following 12 features:
    - season (1: winter, 2: spring, 3: summer, 4: autumn)
    - year (0: 2011, 1: 2012)
    - month (1-12)
    - hour (0-23)
    - holiday (binary indicator)
    - day of week (0: Sun, 1: Mon, ..., 6: Sat)
    - working day (binary indicator; a day is neither weekend nor holiday)
    - weather (1: clear, 2: mist, 3: light rain, 4: heavy rain)
    - temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1)
    - feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1)
    - relative humidity (0-1 range)
    - windspeed (normalized to 0-1 range)

    The target variable is the number of rentals in the given hour.
    """
    def __init__(self,
                 name="rental_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name))
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="rental_competition_pipe.model", type=str, help="Model path")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--test_size", default=0.15, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")



def main(args: argparse.Namespace):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # feature engineering
        cols_int = [0, 2, 3, 5, 7]  # one-hot features
        cols_non_int = [4, 5, 6, 7, 8, 9, 10, 11]     # poly features

        train_data = train.data
        train_target = train.target

        pipe = sklearn.pipeline.Pipeline([('col-trans-hot-std', sklearn.compose.ColumnTransformer(
                                                           [("one-hot"  , sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown="ignore"), cols_int),
                                                            ("scale_std", sklearn.preprocessing.StandardScaler(), cols_non_int)])),
                     ('poly_features', sklearn.preprocessing.PolynomialFeatures(2, include_bias=False))])

        train_data = pipe.fit_transform(train_data, train_target)
        # test_data  = pipe.transform(test_data)

        # print(train_data.shape)
        # print(test_data.shape)
        # np.savez("rental_competition.train_test", test_data)


        # TODO: Train a model on the given dataset and store it in `model`.
        
        # testing model for different lambdas
        # lambdas = np.geomspace(0.01, 200, num=100)
        # rmses = []
        # for l in lambdas:
        #     model = sklearn.linear_model.Ridge(alpha = l) 
        #     model.fit(train_data, train_target)
        #     y_pred = model.predict(test_data)              
        #     rmse = np.sqrt( np.sum( (y_pred - test_target)**2 )/len(y_pred) )  
        #     rmses.append(rmse)

        # best_rmse = np.min(rmses)
        # best_lambda = lambdas[np.argmin(rmses)]

        # with open("best.txt", "ab") as f:
        #     f.write(b"\n\n")
        #     np.savetxt(f, np.array([best_rmse, best_lambda, args.test_size])  )
        #     np.savetxt(f, cols_int , fmt='%i')
        #     f.write(b"\n")
        #     np.savetxt(f, cols_non_int , fmt='%i')
        #     f.write(b"-----")

        # if args.plot:
        #     import matplotlib.pyplot as plt
        #     plt.plot(lambdas, rmses)
        #     plt.xscale("log")
        #     plt.xlabel("L2 regularization strength")
        #     plt.ylabel("RMSE")
        #     if args.plot is True: plt.show()
        #     else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")


        best_lambda = 9.946895897987149127e+00

        model = sklearn.linear_model.Ridge(alpha = best_lambda)
        model.fit(train_data, train_target)


        # model = None

        # Serialize the model.
        model = (model, pipe) 
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, as either a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model, pipe = pickle.load(model_file)


        test_data = pipe.transform(test.data)

        ## for submission
        predictions = model.predict(test_data)


        # print(predictions)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
