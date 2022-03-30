from cgi import test
import numpy as np
import copy
import torch
import torch_explain as te

from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from torch_explain_b.logic.nn.entropy import explain_class
from lime import lime_tabular
from lime.submodular_pick import SubmodularPick

from text_classifier_len.explanation_comparison.experiments.utils import (
    check_trust_in_model_len,
    default_predict_fn,
    check_trust_in_model_lime,
)
from text_classifier_len.model_evaluation import train_model_without_dataloader
from text_classifier_len.utils import get_single_stratified_split
from text_classifier_len.utils import get_scores


def get_appropriate_probabilities(probs, num_extra_features, num_targets):
    if len(probs.shape) == 1:
        probs = probs.reshape(1, 1, -1)
    if len(probs.shape) == 2:
        probs = probs.reshape(1, *probs.shape)
    if probs.shape[0] != num_extra_features:
        assert (
            probs.shape[0] == 1
        ), "Should either match the number of extra features or be 1 so that we can broadcast it"
        probs = np.tile(probs, (num_extra_features, 1, 1))
    if probs.shape[1] != num_targets:
        assert (
            probs.shape[1] == 1
        ), "Should either match the number of target classes or be 1 so that we can broadcast it"
        probs = np.tile(probs, (1, num_targets, 1))
    if probs.shape[2] != 2:
        assert probs.shape[2] == 1, "It is a binary probability"
        probs = np.tile(probs, (1, 1, 2))
        probs[:, :, 1] = 1 - probs[:, :, 0]
    return probs


def add_artificial_noise(x, y, noise_probs, num_features=1, seed=None, vals=[0, 0.25]):
    if seed is not None:
        np.random.seed(seed)
    noise_probs = get_appropriate_probabilities(noise_probs, num_features, y.shape[-1])

    x2 = copy.deepcopy(x)
    x2 = np.array(x2.todense())

    x2 = np.hstack([x2, np.zeros((x2.shape[0], num_features))])

    target_idx = [np.nonzero(y[:, i])[0] for i in range(y.shape[-1])]

    for i in range(num_features):
        for j, idx in enumerate(target_idx):
            x2[idx, -(i + 1)] = np.random.choice(
                [0, 0.25], idx.shape, p=noise_probs[i, j]
            )

    return x2


def train_model_with_noise(
    x, y, train_noise_probs, test_noise_probs, num_features=1, vals=[0, 0.25], clf=None,
):
    x_test, x_train, y_test, y_train = get_single_stratified_split(x, y, 10)
    x_train_pert = add_artificial_noise(
        x_train, y_train, train_noise_probs, num_features, vals=vals
    )
    x_test_pert = add_artificial_noise(
        x_test, y_test, test_noise_probs, num_features, vals=vals
    )

    if clf is None:
        clf = RandomForestClassifier(n_estimators=30)

    x_train_act, x_val, y_train_act, y_val = get_single_stratified_split(
        x_train_pert, y_train, 10
    )

    clf.fit(x_train_act, y_train_act)

    y_pred = clf.predict(x_val)
    val_scores = get_scores(y_pred, y_val)

    y_pred = clf.predict(x_test_pert)
    test_scores = get_scores(y_pred, y_test)

    return clf, val_scores, test_scores, (x_train_pert, x_test_pert, y_train, y_test)


def get_len_explanations(
    clf,
    x,
    y,
    model_path,
    concept_names=None,
    predict_fn=default_predict_fn,
    model=None,
    device=torch.device("cpu"),
    batch_size=128,
    learning_rate=1,
    num_epochs=300,
    save_the_model=True,
    loss_func=None,
    n_splits=10,
    learning_rate_scheduler_params=None,
    n_cv_iters=1,
    history_file_path=None,
    weight_reset_module_list=[te.nn.logic.EntropyLinear, torch.nn.Linear],
    optimizer_params=dict(),
    max_minterm_complexity=15,
    topk_explanations=10,
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

    y_target = predict_fn(clf, x)

    model, tot_history = train_model_without_dataloader(
        model,
        csr_matrix(x),
        y_target,
        device,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        save_the_model=save_the_model,
        model_path=model_path,
        loss_func=loss_func,
        n_splits=n_splits,
        learning_rate_scheduler_params=learning_rate_scheduler_params,
        n_cv_iters=n_cv_iters,
        history_file_path=history_file_path,
        weight_reset_module_list=weight_reset_module_list,
        optimizer_params=optimizer_params,
    )

    with open("{}_0".format(model_path), "rb") as f:
        model.load_state_dict(torch.load(f))

    # TODO: Check if the below line is needed
    model.add_module("{}".format(sum(1 for _ in model.children())), torch.nn.Sigmoid())

    yt = 1 * (y_target > 0.5)
    xt, xv, yt, yv = get_single_stratified_split(x, yt, 10, 0)

    xt = torch.FloatTensor(xt)
    yt = torch.FloatTensor(yt)

    xv = torch.FloatTensor(xv)
    yv = torch.FloatTensor(yv)

    org_exps = []
    imp_exps = []
    for i in range(yt.size(-1)):
        org_exps.append(
            te.logic.entropy.explain_class(
                model,
                xt,
                yt,
                xv,
                yv,
                target_class=i,
                max_minterm_complexity=max_minterm_complexity,
                concept_names=concept_names,
                topk_explanations=topk_explanations,
            )
        )
        imp_exps.append(
            explain_class(
                model,
                xt,
                yt,
                xv,
                yv,
                target_class=i,
                max_minterm_complexity=max_minterm_complexity,
                concept_names=concept_names,
                topk_explanations=topk_explanations,
                try_all=True,
            )
        )

    return model, org_exps, imp_exps


def get_lime_explanations(
    clf,
    x,
    y,
    concept_names,
    tag_names,
    discretize_continuous=False,
    sample_size=10,
    num_features=5,
    num_exps_desired=5,
    predictor=default_predict_fn,
    method="sample",
    num_samples=5000,
):
    explainer = lime_tabular.LimeTabularExplainer(
        x,
        feature_names=concept_names,
        class_names=tag_names,
        discretize_continuous=discretize_continuous,
    )
    predict_fn = lambda inp: predictor(clf, inp)
    exp = SubmodularPick(
        explainer,
        x,
        predict_fn=predict_fn,
        sample_size=sample_size,
        num_features=num_features,
        num_exps_desired=num_exps_desired,
        method=method,
        num_samples=num_samples,
    )
    return explainer, exp


def run_single_experiment(
    x,
    y,
    train_noise_probs,
    test_noise_probs,
    model_path,
    concept_names,
    tag_names,
    num_features=1,
    vals=[0, 0.25],
    clf=None,
    discretize_continuous=False,
    lime_sample_size=10,
):
    (
        clf,
        val_scores,
        test_scores,
        (x_train_pert, x_test_pert, y_train, y_test),
    ) = train_model_with_noise(
        x,
        y,
        train_noise_probs=train_noise_probs,
        test_noise_probs=test_noise_probs,
        num_features=num_features,
        vals=vals,
        clf=clf,
    )
    if val_scores[1] - test_scores[1] < 0.03:
        return None

    if len(concept_names) == x.shape[-1]:
        # Add names for the noises
        concept_names.extend(
            ["Noise Feature ({})".format(i + 1) for i in range(num_features)]
        )

    untrustworthy_idx = np.arange(x.shape[-1], x.shape[-1] + num_features)

    print("--- Getting LEN Explanations ---")
    model, org_len_exp, imp_len_exp = get_len_explanations(
        clf, x_train_pert, y_train, model_path=model_path, concept_names=concept_names
    )
    org_len_trust = True
    for exp, _ in org_len_exp:
        org_len_trust = check_trust_in_model_len(exp, concept_names, untrustworthy_idx)
        if not org_len_trust:
            break
    imp_len_trust = True
    for exp, _ in imp_len_exp:
        imp_len_trust = check_trust_in_model_len(exp, concept_names, untrustworthy_idx)
        if not imp_len_trust:
            break

    print("--- Getting LIME Explanations ---")
    if not isinstance(discretize_continuous, list):
        discretize_continuous = [discretize_continuous]
    lime_exps = []
    for dc in discretize_continuous:
        inp_idx = np.arange(x_train_pert.shape[0])
        np.random.shuffle(inp_idx)
        inp_idx = inp_idx[:lime_sample_size]
        inp = x_train_pert[inp_idx]

        explainer, lime_exp = get_lime_explanations(
            clf,
            inp,
            y_train[inp_idx],
            concept_names=concept_names,
            tag_names=tag_names,
            discretize_continuous=dc,
            method="full",
        )

        trusts = []
        for i, explanation in enumerate(lime_exp.sp_explanations):
            for target in explanation.available_labels():
                trusts.append(
                    check_trust_in_model_lime(
                        explainer,
                        explanation,
                        inp[i].reshape(1, -1),
                        target,
                        untrustworthy_idx,
                    )
                )

        lime_exps.append((lime_exp, explainer, x_train_pert[inp_idx], trusts))

    inp_idx = np.arange(x_train_pert.shape[0])
    np.random.shuffle(inp_idx)
    inp_idx = inp_idx[:250]
    inp = x_train_pert[inp_idx]

    explainer, lime_exp = get_lime_explanations(
        clf,
        inp,
        y_train[inp_idx],
        concept_names=concept_names,
        tag_names=tag_names,
        discretize_continuous=True,
        method="full",
        num_samples=20,
    )

    trusts = []
    for i, explanation in enumerate(lime_exp.sp_explanations):
        for target in explanation.available_labels():
            trusts.append(
                check_trust_in_model_lime(
                    explainer,
                    explanation,
                    inp[i].reshape(1, -1),
                    target,
                    untrustworthy_idx,
                )
            )

    more_lime = (lime_exp, explainer, x_train_pert[inp_idx], trusts)

    return (
        clf,
        val_scores,
        test_scores,
        (org_len_exp, model, org_len_trust),
        (imp_len_exp, model, imp_len_trust),
        lime_exps,
        more_lime
    )

