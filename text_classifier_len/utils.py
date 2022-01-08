import numpy as np
import seaborn as sns
import torch

from sklearn import metrics
from scipy import sparse
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from matplotlib import pyplot as plt
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
from torch_explain.nn import EntropyLinear
from torch_explain_b.logic.nn.entropy import _local_explanation


def avg_jaccard(y_true, y_pred):
    """
    see https://en.wikipedia.org/wiki/Multi-label_classification#Statistics_and_evaluation_metrics
    """
    jaccard = np.minimum(y_true, y_pred).sum(axis=1) / np.maximum(y_true, y_pred).sum(
        axis=1
    )

    return jaccard.mean() * 100


def print_score(y_pred, y_true):
    print(
        "F-Beta score (Beta = 0.5): {}".format(
            metrics.fbeta_score(y_true, y_pred, beta=0.5, average="weighted")
        )
    )
    print("F1 score: {}".format(metrics.f1_score(y_true, y_pred, average="weighted")))
    print("Jaccard score: {}".format(avg_jaccard(y_true, y_pred)))
    print("Hamming loss: {}".format(metrics.hamming_loss(y_true, y_pred) * 100))
    print(
        "Precision score: {}".format(
            metrics.precision_score(y_true, y_pred, average="weighted")
        )
    )
    print(
        "Recall score: {}".format(
            metrics.recall_score(y_true, y_pred, average="weighted")
        )
    )
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


def get_learning_curves(history, epoch_limit=None):
    sns.set_theme(
        style="whitegrid", rc={"figure.figsize": (11.7, 8.27), "grid.linestyle": "--"}
    )

    training = [t for (_, _, t) in history]
    learning_rate = [l[0] for (_, l, _) in history]
    validation = [v for (v, _, _) in history]

    new_col = sns.color_palette("tab10")
    temp = new_col[0]
    new_col[0] = new_col[1]
    new_col[1] = temp

    fig = plt.figure()

    host = fig.add_axes([1, 1, 1, 1], axes_class=HostAxes)
    par1 = ParasiteAxes(host, sharex=host)
    par2 = ParasiteAxes(host, sharex=host)
    host.parasites.append(par1)
    host.parasites.append(par2)

    host.axis["right"].set_visible(False)

    par1.axis["right"].set_visible(True)
    par1.axis["right"].major_ticklabels.set_visible(True)
    par1.axis["right"].label.set_visible(True)

    par2.axis["right2"] = par2.new_fixed_axis(loc="right", offset=(60, 0))

    sns.lineplot(
        data=validation,
        color=new_col[0],
        dashes=False,
        label="Validation Score",
        ax=host,
        legend=False,
    )
    tm = max(training)
    sns.lineplot(
        data=[tm - t for t in training],
        color=new_col[1],
        dashes=False,
        label="Training Score",
        ax=par1,
        legend=False,
    )
    sns.lineplot(
        data=learning_rate,
        color=new_col[3],
        dashes=False,
        label="Learning Rate",
        ax=par2,
        legend=False,
    )

    if epoch_limit is not None:
        plt.axvline(epoch_limit)
        text_y = (min(validation) + max(validation)) / 2
        text_y *= 0.90
        plt.text(x=epoch_limit + 10, y=text_y, s="Epoch Limit", rotation=90)

    host.set_xlim(left=0, right=len(validation))

    host.set_xlabel("Epoch")
    host.set_ylabel("Validation Score")
    par1.set_ylabel("Training Score (Max Loss - Current Loss)")
    par2.set_ylabel("Learning Rate")

    host.legend(loc="center right")

    host.axis["left"].label.set_color(new_col[0])
    par1.axis["right"].label.set_color(new_col[1])
    par2.axis["right2"].label.set_color(new_col[3])

    # ti = plt.xticks()
    step = 50
    plt.xticks(np.arange(0, len(validation) + step, step))

    plt.show()


def local_explanation(
    model, x, target_class, module=None, feature_names=None, max_minterm_complexity=50
):
    if module is None:
        for mod in model.children():
            if isinstance(mod, EntropyLinear):
                # Just taking the first EntropyLinear Layer
                module = mod
                break
    assert isinstance(module, EntropyLinear), "module should be an EntropyLinear Layer"

    if feature_names is None:
        feature_names = [f"feature{j:010}" for j in range(x.size(1))]

    if x.ndim == 1:
        x = x.view(1, -1)

    y = model(x)
    y = y.view(x.size(0), -1)
    y_correct = y[:, target_class] >= 0.5

    local_explanations = []
    local_explanations_raw = {}

    # look at the "positive" rows of the truth table only
    positive_samples = torch.nonzero(y_correct)
    for positive_sample in positive_samples:
        _, local_explanation_raw = _local_explanation(
            module,
            feature_names,
            positive_sample,
            local_explanations_raw,
            None,
            None,
            target_class=target_class,
            max_accuracy=False,
            max_minterm_complexity=max_minterm_complexity,
            simplify=False,
        )

        local_explanations.append(local_explanation_raw)

    return local_explanations
