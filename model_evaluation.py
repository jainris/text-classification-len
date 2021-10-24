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
from utils import convert_scipy_csr_to_torch_coo


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


class SparseToDenseDataset(torch.utils.data.Dataset):
    def __init__(self, sparse_tensor, tags):
        self.sparse_tensor = sparse_tensor
        self.tags = tags

    def __len__(self):
        return self.sparse_tensor.shape[0]

    def __getitem__(self, index):
        x = self.sparse_tensor[index].to_dense()
        y = self.tags[index]

        return x, y


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


def create_and_train_len(
    x_train, y_train, batch_size=128, learning_rate=1, num_epochs=10
):
    layers = [
        te.nn.EntropyLinear(x_train.shape[1], 10, n_classes=y_train.shape[1]),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(10, 4),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(4, 1),
    ]
    model = torch.nn.Sequential(*layers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_form = torch.nn.BCEWithLogitsLoss()
    model.train()

    training_dataset = SparseToDenseDataset(x_train, torch.FloatTensor(y_train))
    trainig_data_generator = torch.utils.data.DataLoader(
        training_dataset, batch_size=batch_size
    )

    for epoch in range(num_epochs):
        print("Epoch " + str(epoch + 1) + ":")
        i = 0
        for x, y in trainig_data_generator:
            if i == 50:
                # Currently only training with limited samples
                break
            print("Num -- " + str(i))
            i += 1
            optimizer.zero_grad()
            y_pred = model(x).squeeze(-1)
            loss = loss_form(y_pred, y) + 0.0001 * te.nn.functional.entropy_logic_loss(
                model
            )
            loss.backward()
            optimizer.step()

    return model


def test_len(model, x_test, y_test, batch_size=128):
    testing_dataset = SparseToDenseDataset(x_test, torch.FloatTensor(y_test))
    testing_data_generator = torch.utils.data.DataLoader(
        testing_dataset, batch_size=batch_size
    )

    y_preds = []
    y_true = []
    i = 0
    for x, y in testing_data_generator:
        if i == 50:
            # Currently only testing limited samples
            break
        print("Num -- " + str(i))
        i += 1
        y_preds.append(model(x).squeeze(-1))

        y_true.append(y.numpy())

    y_preds = [torch.nn.Sigmoid()(yy).detach().numpy() for yy in y_preds]
    y_preds = np.stack(y_preds)
    y_preds = y_preds.reshape((y_preds.shape[0] * y_preds.shape[1], 100))

    y_true = np.array(y_true).reshape(y_preds.shape)

    y_preds = np.where(y_preds > 0.5, 1, 0)

    print_score(y_preds, y_true)


def run_len(batch_size=128, learning_rate=1, num_epochs=10):
    print("Obtaining the dataset")
    evaluator = ModelEvaluator()

    x_train = evaluator.x_train
    y_train = evaluator.y_train
    x_test = evaluator.x_test
    y_test = evaluator.y_test

    x_train = convert_scipy_csr_to_torch_coo(x_train)
    x_test = convert_scipy_csr_to_torch_coo(x_test)

    model = create_and_train_len(
        x_train=x_train,
        y_train=y_train,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
    )

    test_len(model=model, x_test=x_test, y_test=y_test, batch_size=batch_size)
