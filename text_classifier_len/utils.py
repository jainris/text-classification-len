import numpy as np
import torch

from sklearn import metrics
from scipy import sparse
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def avg_jaccard(y_true, y_pred):
    """
    see https://en.wikipedia.org/wiki/Multi-label_classification#Statistics_and_evaluation_metrics
    """
    jaccard = np.minimum(y_true, y_pred).sum(axis=1) / np.maximum(y_true, y_pred).sum(
        axis=1
    )

    return jaccard.mean() * 100


def print_score(y_pred, y_true):
    print("F1 score: {}".format(metrics.f1_score(y_true, y_pred, average="weighted")))
    print("Jaccard score: {}".format(avg_jaccard(y_true, y_pred)))
    print("Hamming loss: {}".format(metrics.hamming_loss(y_true, y_pred) * 100))
    print("---")


def convert_scipy_csr_to_torch_coo(csr_matrix: sparse.csr.csr_matrix):
    coo_matrix = csr_matrix.tocoo()

    values = coo_matrix.data
    indices = np.vstack((coo_matrix.row, coo_matrix.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = torch.Size(coo_matrix.shape)

    return torch.sparse.FloatTensor(i, v, shape)


def get_single_stratified_split(X, y, n_splits, random_state=0):
    kfold = MultilabelStratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    train_idx, test_idx = next(kfold.split(X, y))

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
