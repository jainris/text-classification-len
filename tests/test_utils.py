import pytest
import numpy as np

from text_classifier_len.utils import avg_jaccard
from text_classifier_len.utils import convert_scipy_csr_to_torch_coo


@pytest.mark.parametrize("y_pred, y_true", [
    (np.arange(24).reshape((4, 6)), np.ones((4, 6))),
    (np.ones((5, 3)), np.ones((5, 3)))
])
def test_avg_jaccard(y_pred, y_true):
    jaccard_score = avg_jaccard(y_pred=y_pred, y_true=y_true)

    # Expecting a single value
    assert jaccard_score.shape == ()
    # Expecting the value in [0, 100] range
    assert 0 <= jaccard_score <= 100
