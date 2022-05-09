import numpy as np
import pandas as pd
import re
import nltk
import contractions
import scipy

from string import punctuation
from bs4 import BeautifulSoup
from nltk.tokenize import ToktokTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

np.random.seed(seed=0)


class StackSampleDatasetLoader:
    """
    Class to load the StackSample Dataset. It creates a dataframe to handle the
    dataset and for filtering.

    Attributes
    ----------
    merged_df : pandas.DataFrame
        The dataframe containing the loaded dataset.

    Methods
    -------
    merge_questions_and_tags
        Combines the questions and tags dataframes to a single dataframe.

    group_tags_together
        Groups tags for the same question together.

    filter_out_low_scoring_questions
        Filters low scoring questions out
    """

    def __init__(
        self,
        questions_path,
        tag_path,
        encoding="ISO-8859-1",
        filter_dataset=True,
        score_threshold=5,
    ):
        """
        Initializes the object and gets the training and testing data.

        Parameters
        ----------
        questions_path : str
            Path to the StackSample Questions CSV

        tag_path : str
            Path to the StackSample Tags CSV

        encoding : str, optional
            The encoding to be used when reading the CSVs.
            Default value: "ISO-8859-1"

        filter_dataset : bool, optional
            If True, then low scoring questions are removed from the dataframe.
            By default, set to True.

        score_threshold : int, optional
            If filter_dataset it True, then this value is used to determine
            whether a question is low scoring or not. By default, set to 5.
        """
        questions_df = pd.read_csv(questions_path, encoding=encoding)
        tags_df = pd.read_csv(tag_path, encoding=encoding)

        self.merged_df = self.merge_questions_and_tags(
            questions_df=questions_df, tags_df=tags_df
        )

        if filter_dataset:
            self.filter_out_low_scoring_questions(score_threshold=score_threshold)
            self.merged_df = self.merged_df.reset_index(drop=True)

        # Removing score and id columns
        self.merged_df.drop(columns=["Id", "Score"], inplace=True)

    def merge_questions_and_tags(self, questions_df, tags_df):
        """ Combines the questions and tags dataframes to a single dataframe. """
        # First grouping tags of the same question together
        tags_df = self.group_tags_together(tags_df=tags_df)

        # Removing the irrelevant columns from questions dataframe
        # (but keeping the score because it might be used for filtering
        #  and id for merging)
        questions_df.drop(
            columns=["OwnerUserId", "CreationDate", "ClosedDate"], inplace=True
        )

        merged_df = questions_df.merge(tags_df, on="Id")
        return merged_df

    def group_tags_together(self, tags_df):
        """ Groups tags for the same question together. """
        tags_df["Tag"] = tags_df["Tag"].astype(str)
        tags_df = tags_df.groupby("Id")["Tag"].apply(lambda tags: list(tags))
        return pd.DataFrame({"Id": tags_df.index, "Tags": tags_df.values})

    def filter_out_low_scoring_questions(self, score_threshold):
        """ Filters low scoring questions out """
        self.merged_df = self.merged_df[self.merged_df["Score"] > score_threshold]


class DatasetProcessing:
    """
    Class for processing the dataframe.

    Attributes
    ----------
    data_frame : pandas.DataFrame
        The dataset.

    frequent_tags : list
        The list of frequent tags.

    Methods
    -------
    filter_frequent_tags
        Filters the tags to a smaller number

    question_text_processing
        Text processing for the questions
    """

    def __init__(self, data_frame):
        """ Class initilization """
        self.data_frame = data_frame
        self.frequent_tags = None

    def filter_frequent_tags(self, number_of_unique_tags=10):
        """ Filters the tags to a smaller number """
        # Calculating a list of most common tags
        flat_list_of_all_tags = [
            tag
            for list_of_tags in self.data_frame["Tags"].values
            for tag in list_of_tags
        ]
        freq_dist_of_tags = nltk.FreqDist(flat_list_of_all_tags)
        self.frequent_tags = [
            tag for (tag, _) in freq_dist_of_tags.most_common(number_of_unique_tags)
        ]

        # Function to filter frequent tags
        def get_frequent_tags(tags):
            new_tags = []
            for tag in tags:
                if tag in self.frequent_tags:
                    new_tags.append(tag)
            if new_tags == []:
                new_tags = None
            return new_tags

        self.data_frame["Tags"] = self.data_frame["Tags"].apply(get_frequent_tags)

        # Filtering will possibly remove all tags for some questions
        # So dropping such questions
        self.data_frame.dropna(subset=["Tags"], inplace=True)

    def question_text_processing(self):
        """ Text processing for the questions """
        print("Processing Body:")
        print("-> Removing HTML Tags from text")
        # Extracting text from HTML in the body
        self.data_frame["Body"] = self.data_frame["Body"].apply(
            lambda x: BeautifulSoup(x, features="html.parser").get_text()
        )

        self.data_frame["Body"] = TextProcessor(
            text=self.data_frame["Body"], frequent_tags=self.frequent_tags
        ).get_processed_text()
        print("\nProcessing Title:")
        self.data_frame["Title"] = TextProcessor(
            text=self.data_frame["Title"], frequent_tags=self.frequent_tags
        ).get_processed_text()


class TextProcessor:
    """
    Class which holds and applies all the text processing methods.

    Attributes
    ----------
    tokenizer : ToktokTokenizer
        Tokenizer.

    frequent_tags : list
        List of the frequent tags.

    processed_text : str
        The processed text.
    """

    def __init__(self, text, frequent_tags):
        """ Initializes the class """
        self.tokenizer = ToktokTokenizer()
        self.frequent_tags = frequent_tags

        text = text.apply(lambda x: str(x))
        print("-> Cleaning text (abbreviations, etc.)")
        text = text.apply(self.clean_text)
        print("-> Removing punctuations")
        text = text.apply(self.clean_puncts)
        print("-> Lemmatizing")
        text = text.apply(self.lemmatize_words)
        print("-> Removing Stop Words")
        text = text.apply(self.remove_stopwords)

        self.processed_text = text

    def get_processed_text(self):
        """ Returns the processed text """
        return self.processed_text

    def clean_text(self, text):
        """ Cleans the text of contractions, etc. """
        text = contractions.fix(text)
        text = text.lower()
        text = re.sub(r"\'scuse", " excuse ", text)
        text = re.sub(r"\'\n", " ", text)
        text = re.sub(r"\'\xa0", " ", text)
        text = re.sub("\s+", " ", text)
        text = text.strip()
        return text

    @staticmethod
    def strip_list(org_list):
        """ Strip spaces for each element in list """
        new_list = [item.strip() for item in org_list]
        return [item for item in new_list if item != ""]

    @staticmethod
    def get_puncts():
        """ Returns list of punctuations """
        # we want to remove all punctuations except dash
        puncts = ""
        for char in punctuation:
            if char != "-":
                puncts += char
        return puncts

    def clean_puncts(self, text):
        """
        Remove punctuations. Care is taken to not strip punctuations if these
        lead to the tag being present in the question.
        """
        words = self.tokenizer.tokenize(text)
        filtered_list = []
        regex = re.compile("[%s]" % re.escape(self.get_puncts()))
        for w in words:
            if w in self.frequent_tags:
                # We don't want to remove the punctuation from this
                filtered_list.append(w)
            elif w[-1] in [".", ",", "!", "?", ";"] and w[:-1] in self.frequent_tags:
                # Special case if the word is ending with a punct
                filtered_list.append(w[:-1])
            else:
                filtered_list.append(regex.sub("", w))

        filtered_list = self.strip_list(filtered_list)
        return " ".join(map(str, filtered_list))

    def lemmatize_words(self, text):
        """ Lemmatizes words """
        lemmatizer = WordNetLemmatizer()
        words = self.tokenizer.tokenize(text)
        listLemma = []
        for w in words:
            # Pos Tag == VERB
            try:
                x = lemmatizer.lemmatize(w, pos="v")
            except LookupError:
                nltk.download("wordnet")
                x = lemmatizer.lemmatize(w, pos="v")
            listLemma.append(x)
        return " ".join(map(str, listLemma))

    def remove_stopwords(self, text):
        """ Removes stop words """
        try:
            stop_words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords")
            stop_words = set(stopwords.words("english"))
        words = self.tokenizer.tokenize(text)
        filtered_list = [w for w in words if w not in stop_words]
        return " ".join(map(str, filtered_list))


class DataPreparer:
    """
    Data prepper, converting text to vectors.
    
    Attributes
    ----------
    data_frame : pandas.DataFrame
        The dataframe containing the processed dataset.

    tag_binarizer : MultiLabelBinarizer
        Multi label binarizer for the tags.

    title_vectorizer : TfidfVectorizer
        TfidfVectorizer for the titles of the questions.

    body_vectorizer : TfidfVectorizer
        TfidfVectorizer for the body of the questions.

    binarized_tags : np.ndarry
        The binarized tags.

    vectorized_questions : scipy.sparse.csr_matrix
        The vectorized questions.
    """

    def __init__(self, data_frame):
        """ Class initialization """
        self.data_frame = data_frame
        self.tag_binarizer = MultiLabelBinarizer()
        self.title_vectorizer = TfidfVectorizer(
            analyzer="word",
            min_df=0.0,
            max_df=1.0,
            strip_accents=None,
            encoding="utf-8",
            preprocessor=None,
            token_pattern=r"(?u)\S\S+",
            max_features=1000,
        )
        self.body_vectorizer = TfidfVectorizer(
            analyzer="word",
            min_df=0.0,
            max_df=1.0,
            strip_accents=None,
            encoding="utf-8",
            preprocessor=None,
            token_pattern=r"(?u)\S\S+",
            max_features=1000,
        )

        self.binarized_tags = self.tag_binarizer.fit_transform(data_frame["Tags"])
        vectorized_title = self.title_vectorizer.fit_transform(data_frame["Title"])
        vectorized_body = self.body_vectorizer.fit_transform(data_frame["Body"])
        self.vectorized_questions = scipy.sparse.csr_matrix(
            scipy.sparse.hstack([vectorized_title, vectorized_body])
        )
