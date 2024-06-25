#!/usr/bin/env python3
import argparse
import dataclasses

import numpy as np

class ArtificialData:
    @dataclasses.dataclass
    class Sentence: 
        """ Information about a single dataset sentence."""
        gold_edits: int # Number of required edits to be performed.
        predicted_edits: int # Number of edits predicted by a model.
        predicted_correct: int # Number of correct edits predicted by a model.
        human_rating: int # Human rating of the model prediction.

    def __init__(self, args: argparse.Namespace):
        generator = np.random.RandomState(args.seed)

        self.sentences = []
        for _ in range(args.data_size):
            gold = generator.poisson(2)
            correct = generator.randint(gold + 1)
            predicted = correct + generator.poisson(0.5)
            human_rating = max(0, int(100 - generator.uniform(5, 8) * (gold - correct) - generator.uniform(8, 13) * (predicted - correct)))
            self.sentences.append(self.Sentence(gold, predicted, correct, human_rating))


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
# parser.add_argument("--bootstrap_samples", default=100, type=int, help="Bootstrap samples")
parser.add_argument("--bootstrap_samples", default=100, type=int, help="Bootstrap samples")
parser.add_argument("--data_size", default=1000, type=int, help="Data set size")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.



def calculate_TP_FP_FN_HR(sentences):

    TP, FP, FN, HR = [], [], [], []
    for s in sentences:
        TP.append(s.predicted_correct)
        FP.append(s.predicted_edits-s.predicted_correct)
        FN.append(s.gold_edits-s.predicted_correct)
        HR.append(s.human_rating)

    TP, FP, FN = sum(TP), sum(FP), sum(FN)  
    HRavg = np.mean(HR)
    return TP, FP, FN, HRavg


def calculate_F_beta(true_positives, false_positives, false_negatives, beta):

    TP, FP, FN = true_positives, false_positives, false_negatives
    F_beta = (TP + beta**2 *TP)/(TP+ FP + beta**2 * (TP+FN))

    return F_beta 

def correlation_Pearson(x,y):

    x_mean, y_mean = np.mean(x), np.mean(y)
    x_var_sqrt  = np.sqrt( np.sum((x - x_mean)**2) )
    y_var_sqrt  = np.sqrt( np.sum((y - y_mean)**2) )

    r = np.sum( (x - x_mean) * (y - y_mean) )/ ( x_var_sqrt * y_var_sqrt)

    return r

def main(args: argparse.Namespace): # -> tuple[float, float]:
    # Create the artificial data
    data = ArtificialData(args)

    # Create `args.bootstrap_samples` bootstrapped samples of the dataset by
    # sampling sentences of the original dataset, and for each compute
    # - average of human ratings
    # - TP, FP, FN counts of the predicted edits
    human_ratings, predictions = [], []
    generator = np.random.RandomState(args.seed)
    for _ in range(args.bootstrap_samples):
        # Bootstrap sample of the dataset
        sentences = generator.choice(data.sentences, size=len(data.sentences), replace=True)
        
        TP, FP, FN, HR_avg = calculate_TP_FP_FN_HR(sentences)
        human_ratings.append(HR_avg)                            # Append the average of human ratings of `sentences` to `humans`.
        predictions.append([TP, FP, FN])                        # Compute TP, FP, FN counts of predicted edits in `sentences`
                                                                 # and append them to `predictions`. 

    # Compute Pearson correlation between F_beta score and human ratings
    # for betas between 0 and 2.
    betas, correlations = [], []
    for beta in np.linspace(0, 2, 201):
        betas.append(beta)

        
        F_betas = []
        for pred in predictions:
            TP, FP, FN = pred
            F_beta = calculate_F_beta(TP, FP, FN, beta)
            F_betas.append(F_beta)

        correlation = correlation_Pearson(np.array(F_betas), np.array(human_ratings))



        # TODO: For each bootstap dataset, compute the F_beta score using
        # the counts in `predictions` and then manually compute the Pearson
        # correlation between the computed scores and `human_ratings`. Append
        # the result to `correlations`.
        correlations.append(correlation)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(betas, correlations)
        plt.xlabel(r"$\beta$")
        plt.ylabel(r"Pearson correlation of $F_\beta$-score and human ratings")
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    # TODO: Assign the highest correlation to `best_correlation` and
    # store corresponding beta to `best_beta`.

    best_beta, best_correlation = betas[np.argmax(correlations)], np.max(correlations)

    return best_beta, best_correlation

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    best_beta, best_correlation = main(args)

    print("Best correlation of {:.3f} was found for beta {:.2f}".format(
        best_correlation, best_beta))
