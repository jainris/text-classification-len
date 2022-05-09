import numpy as np
import pandas as pd

from text_classifier_len.data_processing import StackSampleDatasetLoader
from text_classifier_len.data_processing import DatasetProcessing
from text_classifier_len.data_processing import TextProcessor
from text_classifier_len.data_processing import DataPreparer


def test_stack_sample_dataset_loader():
    stack_sample_loader = StackSampleDatasetLoader(
        questions_path="tests/sample_dataset/Questions.csv",
        tag_path="tests/sample_dataset/Tags.csv",
    )

    assert stack_sample_loader.merged_df.columns.tolist() == ["Title", "Body", "Tags"]


def test_dataset_processing_and_data_preparer():
    def gen_dummy_dataframe():
        return pd.DataFrame(
            [
                ["Test Data and Testing!", "Sample. Question", "ques1"],
                ["Test! Data2", "The samp-le data 2", "ques2"],
            ],
            columns=["Title", "Body", "Tags"],
        )

    data_frame = gen_dummy_dataframe()
    data_processor = DatasetProcessing(data_frame=data_frame)
    data_processor.filter_frequent_tags()
    data_processor.question_text_processing()

    assert data_processor.data_frame.columns.tolist() == ["Title", "Body", "Tags"]

    data_prepper = DataPreparer(data_frame=data_frame)

    # All values should be either 1 or 0
    assert np.logical_or(
        data_prepper.binarized_tags == 1, data_prepper.binarized_tags == 0
    ).all()
    assert data_prepper.vectorized_questions is not None
