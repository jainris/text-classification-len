import numpy as np
import torch


def default_predict_fn(clf, inp):
    predictions = clf.predict_proba(inp)
    predictions = np.hstack(predictions)
    idx = np.arange(1, predictions.shape[-1], 2)
    predictions = predictions[:, idx]
    return predictions


def check_trust_in_model_len(exp, concept_names, untrustworthy_idx):
    for i in untrustworthy_idx:
        if concept_names[i] in exp:
            return False
    return True


def check_trust_in_model_lime(explainer, exp, inp, target, untrustworthy_idx):
    if explainer.discretizer is not None:
        inp = explainer.discretizer.discretize(inp)
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
