import pandas as pd
import numpy as np
from string import punctuation
from wordcloud import STOPWORDS
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
import pickle
from sklearn.model_selection import train_test_split
import optuna
from src.utils.utils import ReadPrepare, Split
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix


class CustomTfidf:
    """
    Creates customized TF-IDF vectorizer.

    Parameters:
    -----------
    max_df : float, optional (default=0.8)
        The maximum document frequency of a term in the corpus. Terms with a higher
        document frequency will be removed.
    min_df : float, optional (default=10)
        The minimum document frequency of a term in the corpus. Terms with a lower
        document frequency will be removed.
    ngram_range : tuple, optional (default=(1,1))
        The range of n-gram sizes to include in the TF-IDF vectorizer.

    Returns:
    --------
    .fit_transform() : csr_matrix
            The sparse matrix of TF-IDF features.
    or

    .save() : tfidf.pickle
        Saves matrix which can be loaded afterwards via .load()
    """

    def __init__(self,
                 max_df: float = 0.8,
                 min_df: float = 10,
                 ngram_range: tuple = (1, 1),
                 max_feat=None,
                 norm='l2'):
        self._tfidf = TfidfVectorizer(ngram_range=ngram_range, max_df=max_df, min_df=min_df,norm=norm,max_features=max_feat)
        self._dump = None

    def fit_transform(self, x: pd.DataFrame, y=None):
        """
        Fits the TF-IDF vectorizer to the input data and transforms it into a sparse matrix of TF-IDF features.

        Parameters:
        -----------
        x : pandas DataFrame
            The input data to fit the TF-IDF vectorizer to and transform into a TF-IDF
            sparse matrix.
        y : ignored

        Returns:
        --------
        csr_matrix
            The sparse matrix of TF-IDF features.
        """
        self._dump = self._tfidf.fit_transform(x)
        return self._dump

    def save(self):
        """
        Saves the fitted TF-IDF vectorizer to a pickle file.

        Returns:
        --------
        tfidf.pickle
        """
        with open("../../data/tfidf.pickle", 'wb') as file:
            pickle.dump(self._dump, file)

    @staticmethod
    def load():
        """
        Loads a fitted TF-IDF vectorizer from a pickle file.

        Returns:
        --------
        object
            The fitted TF-IDF vectorizer.
        """
        with open("../../data/tfidf.pickle", 'rb') as file:
            return pickle.load(file)


class AddingFeatures:
    """
    Adding indirect features to "comment_text" column and compile it with tf-idf features.

    Parameters:
    -----------
    scaler: object, default=MinMaxScaler()
        Scikit-learn scaler object to scale the features.
    indirect_f: list, default=indirect_f_list
        List of strings, containing names of the indirect features to be added to the text data.

    Methods:
    --------
    add(self, input_df: pd.DataFrame) -> pd.DataFrame:
        Adds indirect features to the input DataFrame.

    normalize()???

    stack(self, tfidf: csr_matrix, indirect_features: csr_matrix) -> csr_matrix:
        Stacks the tf-idf and indirect features horizontally to create a single sparse matrix.

    Returns:
    --------
    self : csr_matrix
    """
    indirect_f_list = ['count_word', 'count_unique_word', 'count_letters', 'count_punctuations', 'count_words_upper',
                       'count_words_title', 'count_stopwords', 'mean_word_len', 'word_unique_percent', 'punct_percent']

    def __init__(self,
                 scaler=MinMaxScaler(),
                 indirect_f: list = indirect_f_list):
        self._scaler = scaler
        self._indirect_f = indirect_f

    def create(self, input_df) -> pd.DataFrame:  # 'DataFrame' object has no attribute 'to_frame'!!!
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
        input_df = input_df.to_frame(name='comment_text')
        input_df.loc[:, 'count_word'] = input_df["comment_text"].apply(lambda x: len(str(x).split()))
        input_df.loc[:, 'count_unique_word'] = input_df["comment_text"].apply(
            lambda x: len(set(str(x).split())))
        input_df.loc[:, 'count_letters'] = input_df["comment_text"].apply(lambda x: len(str(x)))
        input_df.loc[:, "count_punctuations"] = input_df["comment_text"].apply(
            lambda x: len([c for c in str(x) if c in punctuation]))
        input_df.loc[:, "count_words_upper"] = input_df["comment_text"].apply(
            lambda x: len([w for w in str(x).split() if w.isupper()]))
        input_df.loc[:, "count_words_title"] = input_df["comment_text"].apply(
            lambda x: len([w for w in str(x).split() if w.istitle()]))
        input_df.loc[:, "count_stopwords"] = input_df["comment_text"].apply(
            lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
        input_df.loc[:, "mean_word_len"] = input_df["comment_text"].apply(
            lambda x: round(np.mean([len(w) for w in str(x).split()]), 2))
        input_df.loc[:, 'word_unique_percent'] = input_df.loc[:, 'count_unique_word'] * 100 / input_df[
            'count_word']
        input_df.loc[:, 'punct_percent'] = input_df.loc[:, 'count_punctuations'] * 100 / input_df[
            'count_word']
        return input_df[self._indirect_f]

    def normalize(self, X, y=None):
        """
        Normalize (scale) indirect features

        Parameters:
        -----------
        x : pandas DataFrame
            The input data.

        Returns:
        --------
        Normalized (scaled) csr_matrix
        """
        scaled_x = self._scaler.fit_transform(X)
        return csr_matrix(scaled_x)

    def stack(self, tfidf, norm_indirect_features):
        """
        Stacking TF-IDF features & indirect features.

        Parameters:
        -----------
        tfidf : pd.DataFrame
            Input TF-IDF embedding.
        indirect_features : pd.DataFrame
            Input DataFrame containing the text data.
        Returns:
        --------
        Extracted features : pd.DataFrame

        """
        return hstack((tfidf, norm_indirect_features))


#  meta-class hardcoded
class Preprocessor:
    """
    Hardcoded meta-class of TFIDF features
    Preprocessor to transform raw text data into a sparse matrix of features.

    Parameters:
    -----------
    n_samples : int


    Returns:
    --------
    df : csr_matrix
        TF-IDF embedding combined with additional features
    """

    def __init__(self, n_samples=100,
                 vectorizer=CustomTfidf()):
        self.n_samples = n_samples
        self.vectorizer = vectorizer
        self.adding_indirect_f = AddingFeatures()

    # def fit(self, x, y=None):
    #     self.vectorizer.fit(x)
    #     self.adding_indirect_f.fit(x)
    #     return self

    # def transform(self, x, y=None):
    #     tfidf_X = self.vectorizer.transform(x)
    #     indirect_sparse_f = self.adding_indirect_f.transform(x)
    #     return hstack((tfidf_X, indirect_sparse_f))

    def fit_transform(self, x, y=None):
        # path
        # path = "../../../../DB's/Toxic_database/tox_train.csv"
        #
        # # ReadPrepare test
        # df = ReadPrepare(path, 1200).data_process()
        #
        # # Split test
        # train_X, train_y = Split(df=df).get_train_data()

        # CustomTfidf
        tfidf = CustomTfidf().fit_transform(train_X)

        # AddingFeatures create
        additional_features = AddingFeatures().create(input_df=train_X)
        # print(additional_features)

        # AddingFeatures normalize
        normalize = AddingFeatures().normalize(additional_features)

        # AddingFeatures stack
        stack = AddingFeatures().stack(tfidf, normalize)
        return stack # -> csr_matrix


if __name__ == '__main__':
    from src.utils.utils import ReadPrepare, Split

    # path
    path = "../../../../DB's/Toxic_database/tox_train.csv"

    # ReadPrepare test
    df = ReadPrepare(path, 1200).data_process()

    # Split test
    train_X, train_y = Split(df=df).get_train_data()
    #
    # # CustomTfidf test
    # tfidf = CustomTfidf().fit_transform(train_X)
    #
    # # AddingFeatures create test
    # additional_features = AddingFeatures().create(input_df=train_X)
    # # print(additional_features)
    #
    # # AddingFeatures normalize test
    # normalize = AddingFeatures().normalize(additional_features)
    #
    # # AddingFeatures stack test
    # stack = AddingFeatures().stack(tfidf, normalize)

    # Preprocessor test
    p = Preprocessor().fit_transform(train_X)
    print(p)
