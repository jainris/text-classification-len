import pickle

from text_classifier_len.model_evaluation import train_model

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

seed = 0

evaluator = None
with open('10tag_eval', 'rb') as f:
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

model, history = train_model(
    model=model,
    x_train=x_train,
    y_train=y_train,
    device=device,
    batch_size=512,
    learning_rate=2,
    num_epochs=300,
    save_the_model=True,
    model_path='10tags_stratified_stepLR',
)
