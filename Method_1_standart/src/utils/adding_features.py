from string import punctuation
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from wordcloud import STOPWORDS


class AddingFeatures:
    """
    Adding indirect features to "comment_text" column and compile it with vectorizer features.

    Parameters:
    -----------
    _scaler: object, default=MinMaxScaler()
        Scikit-learn _scaler object to scale the features.
    indirect_f: list, default=indirect_f_list
        List of strings, containing names of the indirect features to be added to the text data.

    Methods:
    --------
    create(self, input_df: pd.DataFrame) -> pd.DataFrame:
        Adds indirect features to the input DataFrame.

    normalize(self, X, y=None) -> csr_matrix:
        normalizers

    stack(self, tfidf: csr_matrix, indirect_features: csr_matrix) -> csr_matrix:
        Stacks the tf-idf and indirect features horizontally to create a single sparse matrix.

    Returns:
    --------
    self : csr_matrix
    """

    indirect_f_list = [
        "count_word",
        "count_unique_word",
        "count_letters",
        "count_punctuations",
        "count_words_upper",
        "count_words_title",
        "count_stopwords",
        "mean_word_len",
        "word_unique_percent",
        "punct_percent",
    ]

    def __init__(self, scaler=MinMaxScaler(), indirect_f: list = indirect_f_list):
        self._scaler = scaler
        self._indirect_f = indirect_f

    def create(self, input_df) -> pd.DataFrame:
        """
        Create indirect features from the input DataFrame.

        Parameters:
        -----------
        input_df: pd.DataFrame
            Input DataFrame containing the text data.

        Returns:
        --------
        input_df: pd.DataFrame
            DataFrame which contains the indirect features only.
        """
        input_df = input_df.to_frame(name="comment_text")
        input_df.loc[:, "count_word"] = input_df["comment_text"].apply(
            lambda x: len(str(x).split())
        )
        input_df.loc[:, "count_unique_word"] = input_df["comment_text"].apply(
            lambda x: len(set(str(x).split()))
        )
        input_df.loc[:, "count_letters"] = input_df["comment_text"].apply(
            lambda x: len(str(x))
        )
        input_df.loc[:, "count_punctuations"] = input_df["comment_text"].apply(
            lambda x: len([c for c in str(x) if c in punctuation])
        )
        input_df.loc[:, "count_words_upper"] = input_df["comment_text"].apply(
            lambda x: len([w for w in str(x).split() if w.isupper()])
        )
        input_df.loc[:, "count_words_title"] = input_df["comment_text"].apply(
            lambda x: len([w for w in str(x).split() if w.istitle()])
        )
        input_df.loc[:, "count_stopwords"] = input_df["comment_text"].apply(
            lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS])
        )
        input_df.loc[:, "mean_word_len"] = input_df["comment_text"].apply(
            lambda x: round(np.mean([len(w) for w in str(x).split()]), 2)
        )
        input_df.loc[:, "word_unique_percent"] = (
            input_df.loc[:, "count_unique_word"] * 100 / input_df["count_word"]
        )
        input_df.loc[:, "punct_percent"] = (
            input_df.loc[:, "count_punctuations"] * 100 / input_df["count_word"]
        )
        return input_df[self._indirect_f]

    def normalize(self, X, y=None) -> csr_matrix:
        """
        Takes pd.DataFrame from .create() and normalize it.

        Parameters:
        -----------
        X : pd.DataFrame
            additional features only
        y : None

        Returns:
        -------
        Normalized (scaled) csr_matrix
        """
        scaled_x = self._scaler.fit_transform(X)
        return csr_matrix(scaled_x)

    @staticmethod
    def stack(vectorizer, norm_indirect_features):
        """
        Stacking TF-IDF features & indirect features.

        Parameters:
        -----------
        vectorizer : pd.DataFrame
            Input vectorizer embeddings in csr_matrix format.
        norm_indirect_features : pd.DataFrame
            Input csr_matrix containing normalized additional features.

        Returns:
        --------
        Extracted features : pd.DataFrame

        """
        return hstack((vectorizer, norm_indirect_features))


if __name__ == "__main__":
    from Method_1_standart.src.utils.utils import ReadPrepare, Split

    # path
    path = "../../../../../DB's/Toxic_database/tox_train.csv"

    # ReadPrepare test
    df = ReadPrepare(path, 100000).data_process()

    # Split test
    train_X, train_y = Split(df=df).get_train_data()

    # AddingFeatures create test
    additional_features = AddingFeatures().create(input_df=train_X)
    # print(f"additional_features: {additional_features}")
    # print(f"additional_features type: {type(additional_features)}")

    # AddingFeatures normalize test
    normalize = AddingFeatures().normalize(additional_features)
    # print(f"normalize: {normalize}")
    # print(f"normalize type: {type(normalize)}")

    # AddingFeatures stack test
    from Method_1_standart.src.utils.spacy_vectorizer import SpacyVectorTransformer
    from Method_1_standart.src.utils.custom_tf_idf_vectorizer import CustomTfidf

    def run(vectorizer: str = None):
        """
        Params:
        -------
        vectorizer: 'spacy' or 'tfidf'"""
        if vectorizer == "spacy":
            vectorizer = SpacyVectorTransformer().fit_transform(train_X)
        elif vectorizer == "tfidf":
            vectorizer = CustomTfidf().fit_transform(train_X)

        stack = AddingFeatures().stack(vectorizer, normalize)
        n = "\n"
        return f"stack: {stack} {n} type: {type(stack)}"

    # print(run(vectorizer="spacy"))
    X_train = CustomTfidf().fit_transform(train_X)

    # checking SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report

    svc = LogisticRegression()
    svc.fit(X_train, y=train_y)
    pred_y = svc.predict(X_train)
    report = classification_report(train_y, pred_y)
    print(report)
