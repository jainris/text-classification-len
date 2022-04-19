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
    """ Calculates the average Jaccard Index. """
    jaccard = np.minimum(y_true, y_pred).sum(axis=1) / np.maximum(y_true, y_pred).sum(
        axis=1
    )

    return jaccard.mean() * 100


def get_scores(y_pred, y_true):
    """ Calculates all the scores. """
    return [
        metrics.fbeta_score(y_true, y_pred, beta=0.5, average="weighted"),
        metrics.f1_score(y_true, y_pred, average="weighted"),
        avg_jaccard(y_true, y_pred),
        metrics.hamming_loss(y_true, y_pred) * 100,
        metrics.precision_score(y_true, y_pred, average="weighted"),
        metrics.recall_score(y_true, y_pred, average="weighted"),
    ]


def print_score(scores=None, y_pred=None, y_true=None):
    """ Prints the scores """
    if scores is None:
        assert y_pred is not None and y_true is not None
        scores = get_scores(y_pred=y_pred, y_true=y_true)

    fbeta, f1, jaccard, hamming, precision, recall = scores
    print("F-Beta score (Beta = 0.5): {}".format(fbeta))
    print("F1 score: {}".format(f1))
    print("Jaccard score (%): {}".format(jaccard))
    print("Hamming loss (%): {}".format(hamming))
    print("Precision score: {}".format(precision))
    print("Recall score: {}".format(recall))
    print("---")


def convert_scipy_csr_to_torch_coo(csr_matrix: sparse.csr.csr_matrix):
    """ Converts a CSR matrix to COO Tensor """
    coo_matrix = csr_matrix.tocoo()

    values = coo_matrix.data
    indices = np.vstack((coo_matrix.row, coo_matrix.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = torch.Size(coo_matrix.shape)

    return torch.sparse.FloatTensor(i, v, shape)


def get_single_stratified_split(X, y, n_splits, random_state=0):
    """ Returns a single stratified split. Useful for test-train split """
    kfold = MultilabelStratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    train_idx, test_idx = next(kfold.split(X, y))

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def get_learning_curves(history, epoch_limit=None):
    """ Plots the learning curves """
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
    model,
    x,
    target_class,
    module=None,
    feature_names=None,
    max_minterm_complexity=10,
    simplify=False,
    improve=True,
    ignore_improb=True,
):
    """
    Gives the local explanation for a LEN model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be explained.

    x : torch.nn.Tensor
        The input to be explained.

    target_class : int
        The target class which we want explained.

    module : torch.nn.Module, optional
        If not None, then uses this specific module for obtaining explanation.
        Else takes the first module the input encounters.

    feature_names : list, optional
        List containing the names of the input concepts.

    max_minterm_complexity : int, optional
        Maximum number of terms in conjunction. Default: 10.

    simplify : bool, optional
        If True, then simplifies the formula before returning. Default: False.

    improve : bool, optional
        If True, then uses the improved form of explanation extraction.
        Else, uses the original method. Default: True.

    ignore_improb : bool, optional
        If True, then ignores the input if the model does not predict the
        target class for given input x. Default: True.

    local_explanations : list
        List with an explanation for each valid input of x.
    """
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
    if ignore_improb:
        y_correct = y[:, target_class] >= 0.5
    else:
        y_correct = y[:, target_class] >= 0.0 # All True

    local_explanations = []
    local_explanations_raw = {}

    # look at the "positive" rows of the truth table only
    positive_samples = torch.nonzero(y_correct)
    for positive_sample in positive_samples:
        local_explanation, local_explanation_raw = _local_explanation(
            module,
            feature_names,
            positive_sample,
            local_explanations_raw,
            None,
            None,
            target_class=target_class,
            max_accuracy=False,
            max_minterm_complexity=max_minterm_complexity,
            simplify=simplify,
        )

        good, bad = None, None

        if improve:
            good, bad = get_the_good_and_bad_terms(
                model=model,
                input_tensor=x[positive_sample],
                explanation=local_explanation_raw,
                target=target_class,
                concept_names=feature_names,
            )
            good, bad = " & ".join(good), " & ".join(bad)

        local_explanations.append((local_explanation, local_explanation_raw, good, bad))

    return local_explanations


def get_the_good_and_bad_terms(
    model, input_tensor, explanation, target, concept_names=None
):
    """ Divides terms into good terms and bad terms """
    def perturb_inputs_rem(inputs, target):
        inputs[:, target] = 0.0
        return inputs

    def perturb_inputs_add(inputs, target):
        # inputs[:, target] += inputs.sum(axis=1) / (inputs != 0).sum(axis=1)
        inputs[:, target] += inputs.max(axis=1)[0]
        # inputs[:, target] += 1
        return inputs

    input_tensor = input_tensor.view(1, -1)
    explanation = explanation.split(" & ")

    good, bad = [], []

    base = model(input_tensor).view(-1)
    base = base[target]

    for term in explanation:
        atom = term
        remove = True
        if atom[0] == "~":
            remove = False
            atom = atom[1:]

        if concept_names is not None:
            idx = concept_names.index(atom)
        else:
            idx = int(atom[len("feature") :])
        temp_tensor = input_tensor.clone().detach()
        temp_tensor = (
            perturb_inputs_rem(temp_tensor, idx)
            if remove
            else perturb_inputs_add(temp_tensor, idx)
        )
        new_pred = model(temp_tensor).view(-1)
        new_pred = new_pred[target]
        if new_pred >= base:
            bad.append(term)
        else:
            good.append(term)
        del temp_tensor
    return good, bad


def weight_reset(module, module_list):
    """ Reset weights """
    if type(module) in module_list:
        module.reset_parameters()
