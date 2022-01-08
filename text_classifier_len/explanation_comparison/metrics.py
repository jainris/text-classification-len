import numpy as np
import torch

from inspect import signature
from typing import Any, Callable, Tuple, Union
from captum._utils.common import _format_input
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from torch import Tensor

from text_classifier_len.explanation_comparison.utils import default_perturb_func
from text_classifier_len.explanation_comparison.utils import get_attributes
from text_classifier_len.explanation_comparison.utils import len_explanation_func


def calculate_max_sensitivity_len(
    model,
    inputs,
    target,
    n_perturb_samples=5,
    perturb_func: Callable = default_perturb_func,
    perturb_radius: float = 0.02,
    max_minterm_complexity=10,
):
    def calculate_distance(formula1, formula2=None):
        formula1, fa1, lfa1, form_1 = get_attributes(formula1)
        formula2, fa2, lfa2, form_2 = get_attributes(formula2)

        all_atoms = list(fa1.union(fa2))

        mapping1 = np.zeros(len(lfa1), dtype=int)
        mapping2 = np.zeros(len(lfa2), dtype=int)
        for i in range(len(all_atoms)):
            if all_atoms[i] in lfa1:
                idx = lfa1.index(all_atoms[i])
                mapping1[idx] = i
            if all_atoms[i] in lfa2:
                idx = lfa2.index(all_atoms[i])
                mapping2[idx] = i

        num = 0
        for i in range(1 << len(all_atoms)):
            instance = i & (1 << np.arange(len(all_atoms))) > 0
            params_1 = [instance[mapping] for mapping in mapping1]
            params_2 = [instance[mapping] for mapping in mapping2]

            if form_1(*params_1) != form_2(*params_2):
                num += 1
        prob = num / (1 << len(all_atoms))

        return num, prob, 1 << len(all_atoms)

    def _generate_perturbations(
        current_n_perturb_samples: int,
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        The perturbations are generated for each example
        `current_n_perturb_samples` times.

        For perfomance reasons we are not calling `perturb_func` on each example but
        on a batch that contains `current_n_perturb_samples` repeated instances
        per example.
        """
        inputs_expanded: Union[Tensor, Tuple[Tensor, ...]] = tuple(
            torch.repeat_interleave(input, current_n_perturb_samples, dim=0)
            for input in inputs
        )
        if len(inputs_expanded) == 1:
            inputs_expanded = inputs_expanded[0]

        return (
            perturb_func(inputs_expanded, perturb_radius)
            if len(signature(perturb_func).parameters) > 1
            else perturb_func(inputs_expanded)
        )

    def _next_sensitivity_max():
        current_n_perturb_samples = 1
        inputs_perturbed = _generate_perturbations(current_n_perturb_samples)

        expl_perturbed_inputs = len_explanation_func(
            model=model,
            input_tensor=inputs_perturbed,
            target=target,
            max_minterm_complexity=max_minterm_complexity,
        )

        _, sensitivities, _ = calculate_distance(expl_perturbed_inputs, expl_inputs)

        # compute the norm/distance for each input noisy example
        sensitivities_norm = sensitivities / expl_inputs_norm
        return sensitivities_norm

    inputs = inputs.view(1, *inputs.size())
    inputs = _format_input(inputs)  # type: ignore

    with torch.no_grad():
        expl_inputs = len_explanation_func(
            model,
            input_tensor=inputs,
            target=target,
            max_minterm_complexity=max_minterm_complexity,
        )

        # compute the norm/distance of original input explanations
        _, expl_inputs_norm, _ = calculate_distance(expl_inputs)

        if expl_inputs_norm == 0:
            print("Warning!!! Couldn't find an explanation for the given input")
            expl_inputs_norm = 1.0

        metrics_max = _next_sensitivity_max()
        for _ in range(1, n_perturb_samples):
            metrics_max = max(metrics_max, _next_sensitivity_max())
    return metrics_max


def auc_morf(model, inputs, target, get_importance_sorted_inputs):
    def perturb_inputs(inputs, target, remove=True):
        inputs[:, target] = 0.0 if remove else 1.0
        return inputs

    def get_prediction(model, x, target):
        y_preds = model(x)
        batch_size = y_preds.size(0)
        y_preds = y_preds.view(batch_size, -1)
        y_preds = y_preds[:, target]
        return y_preds, batch_size

    def normalize_input(inputs):
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.view(-1)
            return inputs.sum() / inputs.size(0)
        else:
            return inputs

    inputs = inputs.clone()
    if inputs.ndim < 2:
        inputs = inputs.view(1, -1)
    auc_morf = 0.0

    importance_sorted_inputs = get_importance_sorted_inputs(model, inputs, target)
    i, remove = next(importance_sorted_inputs)
    inputs = perturb_inputs(inputs, i, remove)
    y_prev, _ = get_prediction(model, inputs, target)

    for i, remove in importance_sorted_inputs:
        inputs = perturb_inputs(inputs, i, remove)
        y_cur, _ = get_prediction(model, inputs, target)
        auc_morf += normalize_input((y_prev + y_cur) / 2)
        y_prev = y_cur

    return auc_morf


def calculate_avg_auc_morf(model, inputs, get_importance_sorted_inputs):
    with torch.no_grad():
        auc_morfs = []
        if inputs.ndim == 1:
            inputs = inputs.view(1, -1)
        batch_size = inputs.size(0)

        for b in range(batch_size):
            batch_auc_morfs = []
            x = inputs[b]
            y = model(x).view(-1) >= 0.5

            targets = torch.nonzero(y)

            for target in targets:
                batch_auc_morfs.append(
                    auc_morf(model, x, int(target), get_importance_sorted_inputs)
                )

            auc_morfs.append(batch_auc_morfs)
    return np.mean(auc_morfs), auc_morfs
