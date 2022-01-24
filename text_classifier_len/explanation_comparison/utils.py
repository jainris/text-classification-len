import numpy as np
import torch
import sympy

from captum._utils.common import _format_input
from captum.attr import Lime

from torch import Tensor
from torch.nn import Module

from typing import Any, Callable, Generator, List, Optional, Set, Tuple
from torch_explain.logic.nn import entropy

from text_classifier_len.utils import local_explanation


def default_perturb_func(inputs: Tensor, perturb_radius: float = 0.02) -> Tensor:
    inputs = _format_input(inputs)
    perturbed_input = tuple(
        input
        + torch.FloatTensor(input.size())
        .uniform_(-perturb_radius, perturb_radius)
        .to(input.device)
        for input in inputs
    )
    return perturbed_input


def get_attributes(formula: str) -> Tuple[Any, Set, List, Callable]:
    if formula is None:
        return None, set(), list(), lambda *args: False
    formula = sympy.sympify(formula)
    atoms = formula.atoms()
    l_atoms = list(atoms)
    lamb = sympy.lambdify(l_atoms, formula)
    return formula, atoms, l_atoms, lamb


def len_explanation_func(
    model: Module,
    input_tensor: Tensor,
    target: int,
    max_minterm_complexity: int = 10,
    concept_names: Optional[List] = None,
) -> str:
    if isinstance(input_tensor, Tuple):
        assert len(input_tensor) == 1, "Currently only support one input at a time"
        input_tensor = input_tensor[0]
    if input_tensor.ndim == 3:
        assert input_tensor.size(0) == 1, "Currently only support one input at a time"
        input_tensor = input_tensor[0]
    expected_output_tensor = model(input_tensor)
    expected_output_tensor = expected_output_tensor.view(
        expected_output_tensor.size()[0], -1
    )
    expected_output_tensor = (expected_output_tensor > 0.5) * 1.0
    explanation, _ = entropy.explain_class(
        model,
        input_tensor,
        expected_output_tensor,
        input_tensor,
        expected_output_tensor,
        target_class=target,
        max_minterm_complexity=max_minterm_complexity,
        concept_names=concept_names,
    )
    return explanation


def explanation_func_lime(model: Module, inputs: Tensor, target: int) -> Tensor:
    lime_explainer = Lime(model)
    explanation = lime_explainer.attribute(inputs, target=target)
    return explanation.view(-1).detach().cpu().numpy()


def get_importance_sorted_inputs_lime_like(
    model: Module,
    inputs: Tensor,
    target: int,
    explanation_func: Callable[[Module, Tensor, int], Tensor],
) -> Generator[Tuple[int, bool], None, None]:
    explanation = explanation_func(model, inputs, target)
    exp_idx = [(explanation[i], i) for i in range(len(explanation))]
    exp_idx.sort(key=lambda x: np.abs(x[0]))

    for _, i in reversed(exp_idx):
        yield i, explanation[i] > 0.0


def get_importance_sorted_inputs_len(
    model: Module, inputs: Tensor, target: int, max_minterm_complexity: int = 10
) -> Generator[Tuple[int, bool], None, None]:
    def get_importance_from_fol_string(explanation: str) -> List[int]:
        def insert_in_binary(x: int, index: int, val: int) -> int:
            value_before_index = x & ((1 << index) - 1)
            x -= value_before_index
            x = x << 1
            x += value_before_index + (val << index)
            return x

        _, _, l_atoms, form = get_attributes(explanation)

        results = []
        for i in range(1 << len(l_atoms)):
            instance = i & (1 << np.arange(len(l_atoms))) > 0

            results.append(form(*instance))

        results = np.array(results)
        tot_pos = np.sum(1 * results)

        values = np.zeros(inputs.size(-1))

        for index in range(len(l_atoms)):
            idx = np.arange(1 << (len(l_atoms) - 1))
            idx = [insert_in_binary(id, index, 0) for id in idx]
            val = np.sum(1 * results[idx])
            val_idx = int(str(l_atoms[index])[1:])
            values[val_idx] = tot_pos - 2 * val

        return values

    concept_names = ["f{}".format(i) for i in range(inputs.size(-1))]
    explanation = len_explanation_func(
        model=model,
        input_tensor=inputs,
        target=target,
        max_minterm_complexity=max_minterm_complexity,
        concept_names=concept_names,
    )
    explanation = get_importance_from_fol_string(explanation)

    exp_idx = [(explanation[i], i) for i in range(len(explanation))]
    exp_idx.sort(key=lambda x: np.abs(x[0]))

    for _, i in reversed(exp_idx):
        yield i, explanation[i] >= 0.0


def get_importance_sorted_inputs_len_local(
    model: Module, inputs: Tensor, target: int, max_minterm_complexity: int = 5
) -> Generator[Tuple[int, bool], None, None]:
    def explanation_func(
        input_tensor: Tensor, target: int, max_minterm_complexity: int
    ) -> str:
        if isinstance(input_tensor, Tuple):
            assert (
                len(input_tensor) == 1
            ), "Currently only support one perturbed input at a time"
            input_tensor = input_tensor[0]
        if input_tensor.ndim == 2:
            assert (
                input_tensor.size(0) == 1
            ), "Currently only support one perturbed input at a time"

        concept_names = ["f{}".format(i) for i in range(input_tensor.size(-1))]
        _, explanation, _, _ = local_explanation(
            model,
            input_tensor,
            target_class=target,
            feature_names=concept_names,
            max_minterm_complexity=max_minterm_complexity,
        )[0]
        return explanation[0]

    def get_importance_from_fol_string_local(explanation: str) -> List[int]:
        # the FOL is just a minterm, already sorted by importance
        # We just need to extract the order from the string
        values = np.zeros(inputs.size(-1))

        symbols = explanation.split(" & ")
        for i, sym in enumerate(symbols):
            value = len(symbols) - i
            value *= -1.0 if sym[0] == "~" else 1.0
            val_idx = int(str(sym.split("~")[-1])[1:])
            values[val_idx] = value

        return values

    explanation = explanation_func(
        inputs, target, max_minterm_complexity=max_minterm_complexity
    )
    explanation = get_importance_from_fol_string_local(explanation)
    exp_idx = [(explanation[i], i) for i in range(len(explanation))]
    exp_idx.sort(key=lambda x: np.abs(x[0]))

    for _, i in reversed(exp_idx):
        yield i, explanation[i] >= 0.0
