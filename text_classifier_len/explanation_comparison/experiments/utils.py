import numpy as np
import torch

from text_classifier_len.utils import local_explanation


def default_predict_fn(clf, inp):
    predictions = clf.predict_proba(inp)
    predictions = np.hstack(predictions)
    idx = np.arange(1, predictions.shape[-1], 2)
    predictions = predictions[:, idx]
    return predictions


def check_trust_in_model_len(
    model,
    input_array,
    target,
    untrustworthy_idx,
    concept_names,
    max_minterm_complexity=10,
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
    switch_idx = untrustworthy_idx[np.nonzero(input_array[0, untrustworthy_idx] > 0)[0]]
    switch_idx = switch_idx.reshape(-1)
    for i in switch_idx:
        if concept_names[i] in good:
            return False
    return True


def check_trust_in_model_lime(explainer, exp, inp, target, untrustworthy_idx):
    org_ans = exp.intercept[target]
    inp_scaled = (inp - explainer.scaler.mean_) / explainer.scaler.scale_
    for i, val in exp.local_exp[target]:
        org_ans += val * inp_scaled[0, i]
    org_ans = org_ans > 0.5

    pert_ans = exp.intercept[target]
    inp_scaled = (inp - explainer.scaler.mean_) / explainer.scaler.scale_
    inp_scaled[0, untrustworthy_idx] = 0
    for i, val in exp.local_exp[target]:
        i = int(i)
        pert_ans += val * inp_scaled[0, i]
    pert_ans = pert_ans > 0.5

    return pert_ans == org_ans
