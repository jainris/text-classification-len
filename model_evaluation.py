import numpy as np
import pandas as pd
import torch_explain as te
import torch
import sklearn
import scipy

from sklearn.model_selection import train_test_split
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
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from data_processing import StackSampleDatasetLoader
from data_processing import DatasetProcessing
from data_processing import DataPreparer
from utils import avg_jaccard
from utils import print_score


class ModelEvaluator:
    def __init__(self, questions_path, tag_path):
        (
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
        ) = self.get_training_and_testing_data(
            questions_path=questions_path, tag_path=tag_path
        )

    def get_training_and_testing_data(self, questions_path, tag_path):
        stack_sample_loader = StackSampleDatasetLoader(
            questions_path=questions_path, tag_path=tag_path,
        )
        data_frame = stack_sample_loader.merged_df
        data_processor = DatasetProcessing(data_frame=data_frame)
        data_processor.filter_frequent_tags()
        data_processor.question_text_processing()

        data_prepper = DataPreparer(data_frame=data_frame)

        return train_test_split(
            data_prepper.vectorized_questions,
            data_prepper.binarized_tags,
            test_size=0.2,
            random_state=0,
        )

    def evaluate(self, classifier, one_vs_rest=False):
        classifier_name = classifier.__class__.__name__
        if one_vs_rest:
            classifier = OneVsRestClassifier(classifier)

        classifier.fit(self.x_train, self.y_train)
        y_pred = classifier.predict(self.x_test)

        return y_pred


@ignore_warnings(category=ConvergenceWarning)
def run_basic_models():
    print("Obtaining the dataset")
    evaluator = ModelEvaluator()

    CV_svc = sklearn.model_selection.GridSearchCV(
        estimator=OneVsRestClassifier(LinearSVC()),
        param_grid={"estimator__C": [1, 10, 100, 1000]},
        cv=5,
        verbose=10,
        scoring=sklearn.metrics.make_scorer(avg_jaccard, greater_is_better=True),
    )

    classifiers = [
        DummyClassifier(),
        SGDClassifier(),
        LogisticRegression(),
        MultinomialNB(),
        LinearSVC(),
        Perceptron(),
        PassiveAggressiveClassifier(),
    ]
    args = [(clf, True) for clf in classifiers]
    args.extend(
        [(MLPClassifier(), False), (RandomForestClassifier(), False), (CV_svc, False)]
    )

    for arg in args:
        classifier, one_vs_rest = arg
        y_pred = evaluator.evaluate(classifier=classifier, one_vs_rest=one_vs_rest)
        print_score(y_pred=y_pred, y_true=evaluator.y_test)


def run_len():
    print("Obtaining the dataset")
    evaluator = ModelEvaluator()

    x_train = evaluator.x_train
    y_train = evaluator.y_train
    x_test = evaluator.x_test
    y_test = evaluator.y_test

    def convert_scipy_csr_to_torch_coo(csr_matrix: scipy.sparse.csr.csr_matrix):
        coo_matrix = csr_matrix.tocoo()

        values = coo_matrix.data
        indices = np.vstack((coo_matrix.row, coo_matrix.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = torch.Size(coo_matrix.shape)

        return torch.sparse.FloatTensor(i, v, shape)

    x_train = convert_scipy_csr_to_torch_coo(x_train)

    layers = [
        te.nn.EntropyLinear(x_train.shape[1], 10, n_classes=y_train.shape[1]),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(10, 4),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(4, 1),
    ]
    model = torch.nn.Sequential(*layers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_form = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(10):
        print("Epoch " + str(epoch + 1) + ":")
        for i in range(1000):
            print("Num -- " + str(i))
            x = x_train[i].to_dense()
            y = torch.FloatTensor(np.array([y_train[i]]))
            optimizer.zero_grad()
            y_pred = model(x).squeeze(-1)
            loss = loss_form(y_pred, y) + 0.0001 * te.nn.functional.entropy_logic_loss(
                model
            )
            loss.backward()
            optimizer.step()

    y_true = y_test[:1000]

    x_test = convert_scipy_csr_to_torch_coo(x_test)

    y_preds = []
    for i in range(1000):
        print("Num -- " + str(i))
        x = x_test[i].to_dense()
        y_preds.append(model(x).squeeze(-1))

    y_preds = [torch.nn.Sigmoid()(yy).detach().numpy() for yy in y_preds]
    y_preds = np.stack(y_preds)
    y_preds = y_preds.reshape((1000, 100))

    print_score(y_preds, y_true)
