import sklearn
import numpy as np
import torch
import scipy

from scipy import sparse

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


def convert_scipy_csr_to_torch_coo(csr_matrix: sparse.csr.csr_matrix):
    coo_matrix = csr_matrix.tocoo()

    values = coo_matrix.data
    indices = np.vstack((coo_matrix.row, coo_matrix.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = torch.Size(coo_matrix.shape)

    return torch.sparse.FloatTensor(i, v, shape)

