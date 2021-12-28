import pickle

from text_classifier_len.model_evaluation import SparseToDenseDataset

import numpy as np
import pandas as pd
from torch import optim
from torch.nn.modules.loss import BCEWithLogitsLoss
import torch_explain as te
import torch
import scipy

from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from torch_explain.logic.nn import entropy
from lime.lime_text import LimeTextExplainer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from text_classifier_len.data_processing import StackSampleDatasetLoader
from text_classifier_len.data_processing import DatasetProcessing
from text_classifier_len.data_processing import DataPreparer
from text_classifier_len.utils import avg_jaccard
from text_classifier_len.utils import print_score
from text_classifier_len.utils import convert_scipy_csr_to_torch_coo
from text_classifier_len.utils import get_single_stratified_split


class JaccardLoss(torch.nn.Module):
    def __init__(self, include_sigmoid=False):
        self.include_sigmoid = include_sigmoid
        super(JaccardLoss, self).__init__()

    def forward(self, y_pred, y_exp, smooth=1):
        if self.include_sigmoid:
            y_pred = y_pred.sigmoid()

        # #flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)

        intersect = (y_pred * y_exp).sum()
        # Using the fact that
        # Union(A, B) = A + B - Intersect()
        union = (y_pred + y_exp).sum() - intersect

        jaccard_score = (intersect + smooth) / (union + smooth)

        # jaccard_loss = 1 - jaccard_score
        return 1 - jaccard_score


seed = 0

evaluator = None
with open("10tag_eval", "rb") as f:
    evaluator = pickle.load(f)

x_train = evaluator.x_train
y_train = evaluator.y_train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

layers = [
    te.nn.EntropyLinear(x_train.shape[1], 10, n_classes=y_train.shape[1]),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(10, 4),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(4, 1),
]
model = torch.nn.Sequential(*layers)

# if load_model:
#     model.load_state_dict(torch.load(model_path))

model.to(device=device)


def train_model(
    model,
    x_train,
    y_train,
    device,
    batch_size=128,
    learning_rate=5e-3,
    num_epochs=10,
    save_the_model=False,
    model_path=None,
    loss_func=None,
):
    """
    Trains a PyTorch Model.

    Parameters
    ----------
    TODO
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="max",
        factor=0.5,
        patience=15,
        cooldown=0,
        verbose=True,
        min_lr=1e-1,
        threshold=1e-2,
        threshold_mode='abs',
    )
    model.train()

    if loss_func is None:
        loss_form = JaccardLoss(include_sigmoid=True)
        loss_func = lambda y_exp, y_act, model, x: loss_form(
            y_exp, y_act
        ) + 1e-4 * te.nn.functional.entropy_logic_loss(model)
    history = []

    n_splits = 5
    kf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    min_valid_loss = -np.inf

    ep = 0

    for epoch in range(num_epochs // n_splits):
        for train_idx, val_idx in kf.split(x_train, y_train):
            ep += 1
            training_dataset = SparseToDenseDataset(
                convert_scipy_csr_to_torch_coo(x_train[train_idx]),
                torch.FloatTensor(y_train[train_idx]),
                device,
            )
            trainig_data_generator = torch.utils.data.DataLoader(
                training_dataset, batch_size=batch_size
            )

            validation_dataset = SparseToDenseDataset(
                convert_scipy_csr_to_torch_coo(x_train[val_idx]),
                torch.FloatTensor(y_train[val_idx]),
                device,
            )
            validation_data_generator = torch.utils.data.DataLoader(
                validation_dataset, batch_size=batch_size
            )

            last_run_loss = 0.0
            running_loss = 0.0
            tot_loss = 0.0
            print("Epoch {}".format(epoch + 1), end="\r")
            for i, data in enumerate(trainig_data_generator, 0):
                x, y = data
                # if i == 50:
                #     # Currently only training with limited samples
                #     break
                optimizer.zero_grad()
                y_pred = model(x).squeeze(-1)
                loss = loss_func(y_pred, y, model, x)
                loss.backward()
                optimizer.step()
                print(
                    "\rEpoch, Batch: [{}, {}] -- Loss: {}".format(
                        ep, i + 1, running_loss / (i + 1),
                    ),
                    end="",
                )

                running_loss += loss.item()
                tot_loss += loss.item()
                if i % 50 == 49:  # print every 50 batches
                    print(
                        "\rEpoch, Batch: [{}, {}] -- Loss: {}".format(
                            ep, i + 1, running_loss / 50
                        ),
                        end="",
                    )
                    last_run_loss = running_loss
                    running_loss = 0.0

            # Validation loop
            valid_loss = 0.0
            num_val = 0
            for x, y in validation_data_generator:
                num_val += 1

                y_pred = model(x).squeeze(-1)
                loss = loss_func(y_pred, y, model, x)

                y_true = y.cpu().numpy()

                y_pred = model(x).squeeze(-1)
                y_pred = torch.nn.Sigmoid()(y_pred).detach().cpu().numpy()

                valid_loss += avg_jaccard(y_true, y_pred)

                print(
                    "\rEpoch: {} -- Last Run Loss: {} -- Validation Score: {}".format(
                        ep, last_run_loss / 50.0, valid_loss / num_val
                    ),
                    end="",
                )

            print(
                "\rEpoch: {} -- Last Run Loss: {} -- Validation Score: {}".format(
                    ep, last_run_loss / 50, valid_loss / num_val
                )
            )

            lr = [float(param_group["lr"]) for param_group in optimizer.param_groups]

            history.append((valid_loss / num_val, lr, tot_loss / len(trainig_data_generator)))
            scheduler.step(valid_loss)

            if save_the_model and valid_loss > min_valid_loss:
                print(
                    "Validation score increased!!! ({} -> {})   Saving the model".format(
                        min_valid_loss / num_val, valid_loss / num_val
                    )
                )
                min_valid_loss = valid_loss

                torch.save(model.state_dict(), model_path)

    return model, history


model, history = train_model(
    model=model,
    x_train=x_train,
    y_train=y_train,
    device=device,
    batch_size=512,
    learning_rate=20,
    num_epochs=2500,
    save_the_model=True,
    model_path="10tags_plateau_jaccard_loss",
)

import pickle

with open("history_plateau_jaccard_loss", "wb") as f:
    pickle.dump(history, f)
