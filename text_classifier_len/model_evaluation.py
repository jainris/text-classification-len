import numpy as np
import pandas as pd
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


class ModelEvaluator:
    """
    Basic class that takes the path to CSVs of questions and tags, obtains the
    data and applies the relevant data processing.

    Attributes
    ----------
    data_prepper : text_classifier_len.data_processing.DataPreparer
        The object that prepares the data.

    x_train : scipy.sparse.csr.csr_matrix
        The training input.

    x_test : scipy.sparse.csr.csr_matrix
        The testing input.

    y_train : np.ndarray
        The expected training output.

    y_test : np.ndarray
        The expected testing output.

    Methods
    -------
    get_training_and_testing_data(
        questions_path, tag_path, number_of_unique_tags=100
    )
        Obtains the data from the given paths and splits it into training and
        testing data.

    evaluate(classifier, one_vs_rest=False)
        Given a sklearn classifier, gives the prediction after fitting the
        classifier.
    """

    def __init__(self, questions_path, tag_path, number_of_unique_tags=100):
        """
        Initializes the object and gets the training and testing data.
        """
        self.data_prepper = None
        (
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
        ) = self.get_training_and_testing_data(
            questions_path=questions_path,
            tag_path=tag_path,
            number_of_unique_tags=number_of_unique_tags,
        )

    def get_training_and_testing_data(
        self, questions_path, tag_path, number_of_unique_tags=100
    ):
        """
        Obtains the data from the given paths and splits it into training and
        testing data.

        Parameters
        ----------
        questions_path : string
            The path to the Questions CSV.

        tag_path : string
            The path to the Tags CSV.

        number_of_unique_tags : int, optional
            The number of unique tags to be taken. The most frequent tags are
            taken. By default set to 100.

        Returns
        -------
        x_train : scipy.sparse.csr.csr_matrix
            The training input.

        x_test : scipy.sparse.csr.csr_matrix
            The testing input.

        y_train : np.ndarray
            The expected training output.

        y_test : np.ndarray
            The expected testing output.
        """
        stack_sample_loader = StackSampleDatasetLoader(
            questions_path=questions_path, tag_path=tag_path,
        )
        data_frame = stack_sample_loader.merged_df
        data_processor = DatasetProcessing(data_frame=data_frame)
        data_processor.filter_frequent_tags(number_of_unique_tags=number_of_unique_tags)
        data_processor.question_text_processing()

        self.data_prepper = DataPreparer(data_frame=data_frame)

        return get_single_stratified_split(
            self.data_prepper.vectorized_questions,
            self.data_prepper.binarized_tags,
            n_splits=5,
            random_state=seed,
        )

    def evaluate(self, classifier, one_vs_rest=False):
        """
        Given a sklearn classifier, gives the prediction after fitting the
        classifier.

        Parameters
        ----------
        classifier
            A sklearn classifier.

        one_vs_rest : bool
            If True, then creates a OneVsRestClassifier based on the given
            classifier. Else, uses the classifier as given.

        Returns
        -------
        y_pred : np.ndarray (TODO Check the exact type)
            The prediction as obtained from the classifier.
        """
        if one_vs_rest:
            classifier = OneVsRestClassifier(classifier)

        classifier.fit(self.x_train, self.y_train)
        y_pred = classifier.predict(self.x_test)

        return y_pred


class SparseToDenseDataset(torch.utils.data.Dataset):
    def __init__(self, sparse_tensor, tags, device):
        self.sparse_tensor = sparse_tensor
        self.tags = tags
        self.device = device

    def __len__(self):
        return self.sparse_tensor.shape[0]

    def __getitem__(self, index):
        x = self.sparse_tensor[index].to_dense().to(self.device)
        y = self.tags[index].to(self.device)

        return x, y


@ignore_warnings(category=ConvergenceWarning)
def run_basic_models(questions_path, tag_path):
    print("Obtaining the dataset")
    evaluator = ModelEvaluator(questions_path=questions_path, tag_path=tag_path)

    CV_svc = GridSearchCV(
        estimator=OneVsRestClassifier(LinearSVC()),
        param_grid={"estimator__C": [1, 10, 100, 1000]},
        cv=5,
        verbose=10,
        scoring=metrics.make_scorer(avg_jaccard, greater_is_better=True),
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
        print("Clf: {}".format(classifier.__class__.__name__))
        print_score(y_pred=y_pred, y_true=evaluator.y_test)


def create_and_train_len(
    x_train,
    y_train,
    batch_size=128,
    learning_rate=5e-3,
    num_epochs=10,
    save_the_model=False,
    model_path=None,
    use_cuda=True,
    load_model=False,
):
    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")

    layers = [
        te.nn.EntropyLinear(x_train.shape[1], 10, n_classes=y_train.shape[1]),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(10, 4),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(4, 1),
    ]
    model = torch.nn.Sequential(*layers)

    if load_model:
        model.load_state_dict(torch.load(model_path))

    model.to(device=device)

    model = train_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        device=device,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        save_the_model=save_the_model,
        model_path=model_path,
    )

    # Adding a Sigmoid layer
    model.add_module(sum(1 for _ in model.children()), torch.nn.Sigmoid())

    return model


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
):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_form = torch.nn.BCEWithLogitsLoss()
    model.train()

    n_splits = 5
    kf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    min_valid_loss = -np.inf

    ep = 0

    for epoch in range(num_epochs // n_splits):
        for train_idx, val_idx in kf.split(x_train):
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
            print("Epoch {}".format(epoch + 1), end="\r")
            for i, data in enumerate(trainig_data_generator, 0):
                x, y = data
                # if i == 50:
                #     # Currently only training with limited samples
                #     break
                optimizer.zero_grad()
                y_pred = model(x).squeeze(-1)
                loss = loss_form(
                    y_pred, y
                ) + 1e-4 * te.nn.functional.entropy_logic_loss(model)
                loss.backward()
                optimizer.step()
                print(
                    "\rEpoch, Batch: [{}, {}] -- Loss: {}".format(
                        ep, i + 1, running_loss / (i + 1),
                    ),
                    end="",
                )

                running_loss += loss.item()
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
                loss = loss_form(
                    y_pred, y
                ) + 1e-4 * te.nn.functional.entropy_logic_loss(model)

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

            if save_the_model and valid_loss > min_valid_loss:
                print(
                    "Validation score increased!!! ({} -> {})   Saving the model".format(
                        min_valid_loss / num_val, valid_loss / num_val
                    )
                )
                min_valid_loss = valid_loss

                torch.save(model.state_dict(), model_path)

    return model


def test_torch_model(model, x_test, y_test, batch_size=128, device="cpu"):
    testing_dataset = SparseToDenseDataset(
        convert_scipy_csr_to_torch_coo(x_test), torch.FloatTensor(y_test), device=device
    )
    testing_data_generator = torch.utils.data.DataLoader(
        testing_dataset, batch_size=batch_size
    )

    y_preds = []
    y_true = []
    i = 0
    for x, y in testing_data_generator:
        # if i == 50:
        #     # Currently only testing limited samples
        #     break
        i += 1
        print("\rBatch: [{}]".format(i), end="")
        y_preds.append(model(x).squeeze(-1).detach().cpu())

        y_true.append(y.cpu().numpy())

    print("")

    # y_preds = [torch.nn.Sigmoid()(yy).detach().cpu().numpy() for yy in y_preds]
    y_preds = [yy.detach().cpu().numpy() for yy in y_preds]

    y_preds = y_preds[:-1]
    y_true = y_true[:-1]

    y_preds = np.stack(y_preds)
    y_preds = y_preds.reshape((y_preds.shape[0] * y_preds.shape[1], -1))

    y_true = np.array(y_true).reshape(y_preds.shape)

    y_preds = np.where(y_preds > 0.5, 1, 0)

    print_score(y_preds, y_true)


def get_len_explanation(
    model, x_train, y_train, evaluator, max_minterm_complexity=10,
):
    # Training-Validation Split
    x_train, x_val, y_train, y_val = get_single_stratified_split(
        x_train, y_train, n_splits=5, random_state=seed
    )

    def get_tensors(x, y):
        return convert_scipy_csr_to_torch_coo(x).to_dense(), torch.FloatTensor(y)

    input_tensor, expected_output_tensor = get_tensors(x_train, y_train)
    validation_input_tensor, validation_output_tensor = get_tensors(x_val, y_val)

    concept_names = [
        name + " (title)"
        for name in list(
            evaluator.data_prepper.title_vectorizer.get_feature_names_out()
        )
    ] + [
        name + " (body)"
        for name in list(evaluator.data_prepper.body_vectorizer.get_feature_names_out())
    ]
    tags = evaluator.data_prepper.tag_binarizer.classes_

    for i in range(expected_output_tensor.size()[-1]):
        explanation, _ = entropy.explain_class(
            model,
            input_tensor,
            expected_output_tensor,
            validation_input_tensor,
            validation_output_tensor,
            target_class=i,
            concept_names=concept_names,
            max_minterm_complexity=max_minterm_complexity,
        )
        print("{}: {}".format(tags[i], explanation))


def run_len(
    questions_path, tag_path, batch_size=128, learning_rate=5e-2, num_epochs=10
):
    print("Obtaining the dataset")
    evaluator = ModelEvaluator(questions_path=questions_path, tag_path=tag_path)

    x_train = evaluator.x_train
    y_train = evaluator.y_train
    x_test = evaluator.x_test
    y_test = evaluator.y_test

    model = create_and_train_len(
        x_train=x_train,
        y_train=y_train,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
    )

    test_torch_model(model=model, x_test=x_test, y_test=y_test, batch_size=batch_size)
    get_len_explanation(
        model=model, x_train=x_train, y_train=y_train, evaluator=evaluator,
    )


def run_lime(
    questions_path,
    tag_path,
    batch_size=128,
    learning_rate=5e-2,
    num_epochs=10,
    save_the_model=False,
    model_path=None,
    idx_to_get_explanation=4,
    labels_for_explanation=[0, 2],
):
    print("Obtaining the dataset")
    evaluator = ModelEvaluator(questions_path=questions_path, tag_path=tag_path)

    x_train = evaluator.x_train
    y_train = evaluator.y_train
    x_test = evaluator.x_test
    y_test = evaluator.y_test

    # Model Creation
    device = torch.device("cuda:0")

    layers = [
        torch.nn.Linear(x_train.shape[1], 256),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(256, y_train.shape[1]),
    ]
    model = torch.nn.Sequential(*layers)

    model.to(device)

    model = train_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        device=device,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        save_the_model=save_the_model,
        model_path=model_path,
    )

    # Adding the Sigmoid Layer
    model.add_module(sum(1 for _ in model.children()), torch.nn.Sigmoid())

    test_torch_model(model, x_test, y_test)

    # Explanation
    concept_names = [
        name + " (title)"
        for name in list(
            evaluator.data_prepper.title_vectorizer.get_feature_names_out()
        )
    ] + [
        name + " (body)"
        for name in list(evaluator.data_prepper.body_vectorizer.get_feature_names_out())
    ]
    tags = evaluator.data_prepper.tag_binarizer.classes_

    explainer = LimeTextExplainer(class_names=tags)
    df = evaluator.data_prepper.data_frame.reset_index(drop=True)
    inp_data = (
        df.iloc[idx_to_get_explanation][0]
        + "*&*&*"
        + df.iloc[idx_to_get_explanation][1]
    )
    y_exp = df.iloc[idx_to_get_explanation][2]

    def run_model(inp_data):
        def conv_string_to_model_input(list_of_inp_data):
            res = []
            for inp_data in list_of_inp_data:
                title, body = inp_data.split("*&*&*")
                title = evaluator.data_prepper.title_vectorizer.transform([title])
                body = evaluator.data_prepper.body_vectorizer.transform([body])
                res.append(scipy.sparse.hstack([title, body]))
            return scipy.sparse.vstack(res)

        inp_data = conv_string_to_model_input(inp_data)
        inp_data = convert_scipy_csr_to_torch_coo(inp_data)
        # return torch.nn.Sigmoid()(model(inp_data)).detach().numpy()
        return model(inp_data).detach().numpy()

    exp = explainer.explain_instance(
        inp_data, run_model, num_features=10, labels=labels_for_explanation
    )

    exp.show_in_notebook(text=False)
