from scipy import sparse
import pytest
import numpy as np

from text_classifier_len.utils import avg_jaccard
from text_classifier_len.utils import print_score
from text_classifier_len.utils import convert_scipy_csr_to_torch_coo


@pytest.mark.parametrize(
    "y_pred, y_true",
    [
        (np.arange(24).reshape((4, 6)), np.ones((4, 6))),
        (np.ones((5, 3)), np.ones((5, 3))),
    ],
)
def test_avg_jaccard(y_pred, y_true):
    jaccard_score = avg_jaccard(y_pred=y_pred, y_true=y_true)

    # Expecting a single value
    assert jaccard_score.shape == ()
    # Expecting the value in [0, 100] range
    assert 0 <= jaccard_score <= 100


@pytest.mark.parametrize("y_shape", [(4, 6), (5, 3)])
def test_print_score(y_shape):
    y_true = np.random.choice([0, 1], size=y_shape)
    y_pred = np.random.choice([0, 1], size=y_shape)

    print_score(y_pred=y_pred, y_true=y_true)


@pytest.mark.parametrize("input_shape", [(4, 6), (5, 3)])
@pytest.mark.parametrize(
    "indices, values",
    [(([0, 1, 2], [0, 1, 2]), [1, 2, 3]), (([0, 2, 2], [1, 1, 2]), [1, 1, 1])],
)
def test_convert_scipy_csr_to_torch_coo(input_shape, indices, values):
    def create_scipy_csr_matrix(shape, indices, values):
        np_array = np.zeros(shape)
        np_array[indices] = values
        return sparse.csr_matrix(np_array)

    csr_matrix = create_scipy_csr_matrix(input_shape, indices, values)

    torch_coo_matrix = convert_scipy_csr_to_torch_coo(csr_matrix=csr_matrix)
    torch_coo_matrix = torch_coo_matrix.coalesce()

    assert list(torch_coo_matrix.shape) == list(input_shape)
    assert torch_coo_matrix.indices().tolist() == list(indices)
    assert torch_coo_matrix.values().tolist() == list(values)
