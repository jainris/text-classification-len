from abc import abstractmethod
from typing import List, Union

import torch
from sklearn.metrics import f1_score, accuracy_score
from torch import nn
import torch.nn.functional as F
from torch.nn import Dropout, Linear
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np
import pytorch_lightning as pl

from torch_explain_b.logic.nn.entropy import explain_class
from torch_explain_b.logic.metrics import test_explanation, complexity
from torch_explain_b.nn.functional import entropy_logic_loss
from torch_explain.nn.logic import EntropyLinear


class BaseExplainer(pl.LightningModule):
    def __init__(self, n_concepts: int, n_classes: int, optimizer: str = 'adamw', loss: _Loss = nn.NLLLoss(),
                 lr: float = 1e-2, activation: callable = F.log_softmax,
                 explainer_hidden: list = (10, 10), l1: float = 1e-5):
        super().__init__()
        self.n_classes = n_classes
        self.loss = loss
        self.optmizer = optimizer
        self.lr = lr
        self.activation = activation
        self.model = None
        self.n_concepts = n_concepts
        self.model_hidden = explainer_hidden
        self.l1 = l1
        self.model = None
        self.save_hyperparameters()

    def forward(self, x):
        y_out = self.model(x).squeeze(-1)
        return y_out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_out = self.forward(x)
        if self.loss.__class__.__name__ == 'CrossEntropyLoss':
            loss = self.loss(y_out, y.argmax(dim=1)) + self.l1 * entropy_logic_loss(self.model)
        else:
            loss = self.loss(y_out, y) + self.l1 * entropy_logic_loss(self.model)
        accuracy = _task_accuracy(y_out, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_out = self.forward(x)
        if self.loss.__class__.__name__ == 'CrossEntropyLoss':
            loss = self.loss(y_out, y.argmax(dim=1)) + self.l1 * entropy_logic_loss(self.model)
        else:
            loss = self.loss(y_out, y) + self.l1 * entropy_logic_loss(self.model)
        accuracy = _task_accuracy(y_out, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_out = self.forward(x)
        accuracy = _task_accuracy(y_out, y)
        self.log('test_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return accuracy

    def configure_optimizers(self):
        if self.optmizer == 'adamw':
            return AdamW(self.model.parameters(), lr=self.lr)

    @abstractmethod
    def explain_class(self, x, y, target_class, topk_explanations=3, **kwargs):
        pass


class Explainer(BaseExplainer):
    def __init__(self, n_concepts: int, n_classes: int, optimizer: str = 'adamw', loss: _Loss = nn.CrossEntropyLoss(),
                 lr: float = 1e-2, activation: callable = F.log_softmax, explainer_hidden: list = (8, 3),
                 l1: float = 1e-5, temperature: float = 0.6, conceptizator: str = 'identity_bool'):
        super().__init__(n_concepts, n_classes, optimizer, loss, lr, activation, explainer_hidden, l1)

        self.temperature = temperature
        self.model_layers = []
        self.model_layers.append(EntropyLinear(n_concepts, explainer_hidden[0], n_classes,
                                               temperature, conceptizator=conceptizator))
        self.model_layers.append(torch.nn.LeakyReLU())
        # self.model_layers.append(Dropout())
        for i in range(1, len(explainer_hidden)):
            self.model_layers.append(Linear(explainer_hidden[i - 1], explainer_hidden[i]))
            self.model_layers.append(torch.nn.LeakyReLU())
            # self.model_layers.append(Dropout())

        self.model_layers.append(Linear(explainer_hidden[-1], 1))
        self.model = torch.nn.Sequential(*self.model_layers)

        self.save_hyperparameters()

    def transform(self, dataloader: DataLoader, x_to_bool: int = 0.5, y_to_one_hot: bool = True):
        x_list, y_out_list, y_list = [], [], []
        for i_batch, (x, y) in enumerate(dataloader):
            y_out = self.forward(x.to(self.device))
            x_list.append(x.cpu())
            y_out_list.append(y_out.cpu())
            y_list.append(y.cpu())
        x, y_out, y = torch.cat(x_list), torch.cat(y_out_list), torch.cat(y_list)

        if x_to_bool is not None:
            x = (x.cpu() > x_to_bool).to(torch.float)
        # if y_to_one_hot:
        #     y = F.one_hot(y)
        return x, y_out, y

    def explain_class(self, train_dataloaders: DataLoader, val_dataloaders: DataLoader, test_dataloaders: DataLoader,
                      target_class: Union[int, str] = 'all', concept_names: List = None,
                      topk_explanations: int = 3, max_minterm_complexity: int = None,
                      max_accuracy: bool = False, x_to_bool: int = 0.5,
                      y_to_one_hot: bool = False, verbose: bool = False):

        x_train, y_train_out, y_train_1h = self.transform(train_dataloaders, x_to_bool=x_to_bool)
        x_val, y_val_out, y_val_1h = self.transform(val_dataloaders, x_to_bool=x_to_bool)
        x_test, y_test_out, y_test_1h = self.transform(test_dataloaders, x_to_bool=x_to_bool)

        # model_accuracy = f1_score(y_test_1h.argmax(dim=1), y_test_out.argmax(dim=1))

        if target_class == 'all':
            target_classes = [i for i in range(y_test_1h.size(1))]
        else:
            target_classes = [target_class]

        result_list = []
        exp_accuracy, exp_fidelity, exp_complexity = [], [], []
        for target_class in target_classes:
            class_explanation, explanation_raw = explain_class(self.model.cpu(),
                                                               x_train, y_train_1h,
                                                               x_val, y_val_1h,
                                                               target_class=target_class,
                                                               topk_explanations=topk_explanations,
                                                               max_minterm_complexity=max_minterm_complexity,
                                                               concept_names=concept_names,
                                                               max_accuracy=max_accuracy)
            if class_explanation:
                metric = f1_score
                metric.__setattr__('average', 'macro')
                explanation_accuracy, y_formula = test_explanation(explanation_raw, x_test, y_test_1h, target_class)
                # explanation_fidelity = accuracy_score(y_test_out[:, target_class] > 0.5, y_formula)
                explanation_fidelity = accuracy_score(y_test_out.argmax(dim=1).eq(target_class), y_formula)
                # explanation_fidelity = accuracy_score(y_val_out.argmax(dim=1).eq(target_class), y_formula)
                explanation_complexity = complexity(class_explanation)
            else:
                explanation_accuracy, explanation_fidelity, explanation_complexity = 0, 0, 0
            results = {
                'target_class': target_class,
                'explanation': class_explanation,
                'explanation_accuracy': explanation_accuracy,
                'explanation_fidelity': explanation_fidelity,
                'explanation_complexity': explanation_complexity,
            }
            if verbose:
                print(f'Target class: {target_class}\n\t Results: {results}')
            result_list.append(results)
            exp_accuracy.append(explanation_accuracy)
            exp_fidelity.append(explanation_fidelity)
            exp_complexity.append(explanation_complexity)
        avg_results = {
            'explanation_accuracy': np.mean(exp_accuracy),
            'explanation_fidelity': np.mean(exp_fidelity),
            'explanation_complexity': np.mean(exp_complexity),
            # 'model_accuracy': model_accuracy,
        }
        return avg_results, result_list


def _task_accuracy(y_out, y):
    return y_out.argmax(dim=1).eq(y.argmax(dim=1)).sum() / len(y)
