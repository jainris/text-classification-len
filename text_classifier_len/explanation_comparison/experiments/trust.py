import numpy as np
import torch
import copy


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
