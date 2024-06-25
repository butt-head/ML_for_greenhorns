#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--dataset", default="diabetes", type=str, help="Standard sklearn dataset to load")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.



def all_numbers_integers(array):

    isintarr = []
    for val in list(array):
        isintarr.append(np.equal(np.mod(val, 1), 0))
    
    return np.all(isintarr)



def main(args: argparse.Namespace): #-> tuple[np.ndarray, np.ndarray]:
    dataset = getattr(sklearn.datasets, "load_{}".format(args.dataset))()


    


    # DONE TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.


    train_data, test_data , train_target, test_target = sklearn.model_selection.train_test_split(dataset.data, dataset.target, 
    test_size=args.test_size, random_state=args.seed)


    # TODO: Process the input columns in the following way:
    #
    # - if a column has only integer values, consider it a categorical column
    #   (days in a week, dog breed, ...; in general integer values can also
    #   represent numerical non-categorical values, but we use this assumption
    #   for the sake of an exercise). Encode the values with one-hot encoding
    #   using `sklearn.preprocessing.OneHotEncoder` (note that its output is by
    #   default sparse, you can use `sparse=False` to generate dense output;
    #   also use `handle_unknown="ignore"` to ignore missing values in test set).
    #
    # - for the rest of the columns, normalize their values so that they
    #   have mean 0 and variance 1; use `sklearn.preprocessing.StandardScaler`.
    #
    # In the output, there should be first all the one-hot categorical features,
    # and then the real-valued features. To process different dataset columns
    # differently, you can use `sklearn.compose.ColumnTransformer`.

    # finding only integer categories
    cols_int, cols_non_int  = [], []
    for i in range(dataset.data.shape[1]):
        if all_numbers_integers(dataset.data[:,i]):
            cols_int.append(i)
        else:
            cols_non_int.append(i)

    # determinig categories
    categories_one_hot = []
    for col in cols_int:
        categories_one_hot.append(np.unique(dataset.data[:,col]))
    # print(categories_one_hot)
    

    # print("cols_int:{} :".format(len(cols_int)),cols_int)
    # print("cols_non_int:{} :".format(len(cols_non_int)), cols_non_int)

    ######
    # one hot features
    # encod = sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown="ignore") # auto determination of categories
    # encod = sklearn.preprocessing.OneHotEncoder(categories=categories_one_hot, sparse=False, handle_unknown="ignore") # categories predefined   

    #train_transf_one_hot = encod.fit_transform(train_data[:,cols_int])

    ######
    # features for standard scaling
    # scale_std = sklearn.preprocessing.StandardScaler()
    # train_scale_std = scale_std.fit_transform(train_data[:,cols_non_int])
    # # train_scale_std = scale_std.fit_transform(dataset.data[:,cols_non_int],x)

    # # mean and variance - initial???
    # print("mean: ",scale_std.mean_)  
    # print("var: ",scale_std.var_)


    ## column transformer for transforming columns differently (one-hot features + std scaling)
    ## defined categories
    # ct = sklearn.compose.ColumnTransformer([("one-hot"  , sklearn.preprocessing.OneHotEncoder(categories=categories_one_hot, sparse=False, handle_unknown="ignore"), cols_int),
    #                                         ("scale_std", sklearn.preprocessing.StandardScaler(), cols_non_int)])
    ## auto categories
    # ct = sklearn.compose.ColumnTransformer([("one-hot"  , sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown="ignore"), cols_int),
    #                                         ("scale_std", sklearn.preprocessing.StandardScaler(), cols_non_int)])


    # train_data_trans = ct.fit_transform(train_data)
    # print(train_data_trans.shape)

    # introducing polynomial features
    # poly_features = sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)
    # train_data_trans_poly = poly_features.fit_transform(train_data_trans)



    # test_data_trans = ct.transform(test_data)
    # test_data_trans_poly = poly_features.transform(test_data_trans)


    # train_data = train_data_trans_poly
    # test_data = test_data_trans_poly

    # print(train_data_trans_poly.shape)
    # print(poly_features.powers_)

    # make pipeline for the data - using steps: col-trans-hot-std: - one-hot feature for categorical cols + standardization and scaling for others
    #                                                              - creating of polynomial features
    pipe = sklearn.pipeline.Pipeline([('col-trans-hot-std', sklearn.compose.ColumnTransformer(
                                                           [("one-hot"  , sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown="ignore"), cols_int),
                                                            ("scale_std", sklearn.preprocessing.StandardScaler(), cols_non_int)])),
                     ('poly_features', sklearn.preprocessing.PolynomialFeatures(2, include_bias=False))])


    

    train_data = pipe.fit_transform(train_data)
    test_data  = pipe.transform(test_data)


    # TODO: To the current features, append polynomial features of order 2.
    # If the input values are [a, b, c, d], you should append
    # [a^2, ab, ac, ad, b^2, bc, bd, c^2, cd, d^2]. You can generate such polynomial
    # features either manually, or using
    # `sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)`.

    # TODO: You can wrap all the feature processing steps into one transformer
    # by using `sklearn.pipeline.Pipeline`. Although not strictly needed, it is
    # usually comfortable.

    # TODO: Fit the feature processing steps on the training data.
    # Then transform the training data into `train_data` (you can do both these
    # steps using `fit_transform`), and transform testing data to `test_data`.

    
    # return
    return train_data[:5], test_data[:5]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    # main(args)  # my
    train_data, test_data = main(args)
    for dataset in [train_data, test_data]:
        for line in range(min(dataset.shape[0], 5)):
            print(" ".join("{:.4g}".format(dataset[line, column]) for column in range(min(dataset.shape[1], 140))))




