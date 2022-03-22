import numpy as np
import torch
import torch_explain as te
import copy

from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix
from lime.lime_tabular import LimeTabularExplainer

from text_classifier_len.utils import get_single_stratified_split
from text_classifier_len.utils import local_explanation
from text_classifier_len.model_evaluation import train_model_without_dataloader
from text_classifier_len.explanation_comparison.experiments.utils import (
    default_predict_fn,
    check_trust_in_model_len,
    check_trust_in_model_lime,
)


def get_trust_vals(
    predict_fn, inputs: torch.Tensor, check_trust_in_model, untrustworthy_idx
):
    perturbed_input = copy.deepcopy(inputs)
    perturbed_input[:, untrustworthy_idx] = 0

    original_preds = predict_fn(inputs) > 0.5
    perturbed_preds = predict_fn(perturbed_input) > 0.5

    true_vals = np.vstack(np.nonzero(original_preds)).T
    false_vals_idx = np.random.choice(
        np.arange(original_preds.size), size=2 * true_vals.shape[0], replace=False
    )

    false_samp_idx = false_vals_idx // original_preds.shape[1]
    false_target_idx = false_vals_idx % original_preds.shape[1]

    false_vals_idx = np.vstack([false_samp_idx, false_target_idx]).T
    false_vals = []
    for idx in false_vals_idx:
        if not original_preds[tuple(idx)]:
            false_vals.append(idx)
            if len(false_vals) == true_vals.shape[0]:
                break
    false_vals = np.vstack(false_vals)
    test_vals = np.vstack([true_vals, false_vals])

    exp_trusts = []
    model_trusts = []
    for samp_idx, target in test_vals:
        exp_trusts.append(
            check_trust_in_model(
                inputs[samp_idx].reshape((1, -1)), target, untrustworthy_idx,
            )
        )
        model_trusts.append(
            original_preds[samp_idx][target] == perturbed_preds[samp_idx][target]
        )

    return exp_trusts, model_trusts


def get_len_trust_vals(
    clf,
    x,
    y,
    device,
    concept_names,
    model_path,
    predict_fn,
    trust_inps,
    untrustworthy_idx,
    model=None,
    max_minterm_complexity=10,
    **kwargs
):
    if model is None:
        layers = [
            te.nn.EntropyLinear(x.shape[1], 8, n_classes=y.shape[1]),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(8, 4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(4, 1),
        ]
        model = torch.nn.Sequential(*layers).to(device)

        next(model.children()).conceptizator.threshold = 0.0

    y_target = predict_fn(x)

    model, _ = train_model_without_dataloader(
        model,
        csr_matrix(x),
        y_target,
        device=device,
        save_the_model=True,
        model_path=model_path,
        n_cv_iters=1,
        **kwargs
    )

    with open("{}_0".format(model_path), "rb") as f:
        model.load_state_dict(torch.load(f))

    model.add_module("{}".format(sum(1 for _ in model.children())), torch.nn.Sigmoid())

    def check_len_trust(
        input_array,
        target,
        untrustworthy_idx,
    ):
        input_tensor = torch.FloatTensor(input_array)
        _, _, good, bad = local_explanation(
            model,
            input_tensor,
            target,
            max_minterm_complexity=max_minterm_complexity,
            improve=True,
            ignore_improb=False,
            feature_names=concept_names,
        )[0]
        switch_idx = untrustworthy_idx[
            np.nonzero(input_array[0, untrustworthy_idx] > 0)[0]
        ]
        switch_idx = switch_idx.reshape(-1)
        return check_trust_in_model_len(good, concept_names, switch_idx)

    return (
        model,
        *(
            get_trust_vals(
                predict_fn=predict_fn,
                inputs=trust_inps,
                check_trust_in_model=check_len_trust,
                untrustworthy_idx=untrustworthy_idx,
            )
        ),
    )


def get_lime_trust_vals(
    clf,
    x,
    y,
    predict_fn,
    trust_inps,
    untrustorthy_idx,
    discretize_continuous=False,
    **kwargs
):
    def check_lime_trust(input_array, target, untrustworthy_idx):
        assert len(input_array) == 1
        exp = explainer.explain_instance(input_array[0], predict_fn, [target], **kwargs)
        return check_trust_in_model_lime(
            explainer, exp, input_array, target, untrustworthy_idx
        )

    explainer = LimeTabularExplainer(x, discretize_continuous=discretize_continuous)

    return (
        explainer,
        *get_trust_vals(
            predict_fn=predict_fn,
            inputs=trust_inps,
            check_trust_in_model=check_lime_trust,
            untrustworthy_idx=untrustorthy_idx,
        ),
    )


def run_a_single_experiment_trust(
    x,
    y,
    num_untrustworthy_elements,
    concept_names,
    clf=None,
    train=True,
    predictor=default_predict_fn,
    lime_args=None,
    len_args=None,
    device=torch.device("cpu"),
    discretize_continuous=[True],
):
    if clf is None:
        clf = RandomForestClassifier(n_estimators=30)

    if train:
        clf.fit(x, y)

    predict_fn = lambda inp: predictor(clf, inp)

    untrustworthy_idx = np.random.choice(
        np.arange(x.shape[-1]), size=num_untrustworthy_elements, replace=False
    )

    # Using only a fraction of the data to train the explainers
    _, x_train, _, y_train = get_single_stratified_split(x, y, 5, 0)
    # Using an even smaller fraction of the data to get the trust values
    _, trust_x, _, trust_y = get_single_stratified_split(x, y, 32, 0)

    model_path = "Model_123"
    max_minterm_complexity = 10
    if len_args is None:
        len_args = {
            "batch_size": 128,
            "learning_rate": 1.2,
            "num_epochs": 100,
            "n_splits": 10,
        }
    else:
        if "model_path" in len_args:
            model_path = len_args["model_path"]
            del len_args["model_path"]
        if "max_minterm_complexity" in len_args:
            max_minterm_complexity = len_args["max_minterm_complexity"]
            del len_args["max_minterm_complexity"]

    len_vals = get_len_trust_vals(
        clf=clf,
        x=x_train,
        y=y_train,
        device=device,
        concept_names=concept_names,
        model_path=model_path,
        predict_fn=predict_fn,
        trust_inps=trust_x,
        untrustworthy_idx=untrustworthy_idx,
        max_minterm_complexity=max_minterm_complexity,
        **len_args
    )

    if lime_args is None:
        lime_args = {"num_samples": 200}

    lime_vals = []
    for dc in discretize_continuous:
        lime_vals.append(
            get_lime_trust_vals(
                clf=clf,
                x=x_train,
                y=y_train,
                predict_fn=predict_fn,
                trust_inps=trust_x,
                untrustorthy_idx=untrustworthy_idx,
                discretize_continuous=dc,
                **lime_args
            )
        )

    return (len_vals, lime_vals)

