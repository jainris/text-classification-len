import sklearn
import numpy as np


def avg_jaccard(y_pred, y_true):
    """
    see https://en.wikipedia.org/wiki/Multi-label_classification#Statistics_and_evaluation_metrics
    """
    jaccard = np.minimum(y_true, y_pred).sum(axis=1) / np.maximum(y_true, y_pred).sum(
        axis=1
    )

    return jaccard.mean() * 100


def print_score(y_pred, y_true):
    print("Jacard score: {}".format(avg_jaccard(y_pred, y_true)))
    print("Hamming loss: {}".format(sklearn.metrics.hamming_loss(y_pred, y_true) * 100))
    print("---")
