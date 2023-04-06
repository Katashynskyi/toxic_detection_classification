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
    """

    def __init__(self,
                 max_df: float = 0.8,
                 min_df: float = 10,
                 ngram_range: tuple = (1, 1)):
        self._tfidf = TfidfVectorizer(ngram_range=ngram_range, max_df=max_df, min_df=min_df)
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


class _AddingFeatures:
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

    fit(self, x: pd.Series) -> object:
        Fits the scaler to the indirect features of the input data.

    transform(self, x: pd.Series, y=None) -> csr_matrix:
        Transforms the indirect features of the input data using the fitted scaler.

    fit_transform(self, x: pd.Series, y=None) -> csr_matrix:
        Fits the scaler to the indirect features of the input data and transforms the data.

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

    def add(self, input_df) -> pd.DataFrame:  # 'DataFrame' object has no attribute 'to_frame'!!!
        """
        Adds indirect features to the input DataFrame.

        Parameters:
        -----------
        input_df: pd.DataFrame
            Input DataFrame containing the text data.

        Returns:
        --------
        input_df: pd.DataFrame
            DataFrame containing the indirect features & without "comment_text".
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
#Todo : i'm left here
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
        new_features = self.add(X)
        scaled_x = self._scaler.fit_transform(new_features)
        return csr_matrix(scaled_x)
# Todo :and here
    def stack(self,tfidf, indirect_features):
        """
        Stacking TF-IDF features & indirect features.

        Parameters:
        -----------
        tfidf : pd.DataFrame
            Input DataFrame containing the text data.
        indirect_features : pd.DataFrame
            Input DataFrame containing the text data.
        Returns:
        --------
        Extracted features : pd.DataFrame

        """
        self.normalize(indirect_features)
        return hstack((tfidf, indirect_features))


#  meta-class
class Preprocessor:
    """
    Preprocessor to transform raw text data into a sparse matrix of features.
    This class includes a Tfidf vectorizer to convert the text into a numerical representation,
    as well as a function to add indirect features to the data.
    Parameters:
    -----------
    n_samples:int

    Returns:
    --------
    df : csr_matrix
        TF-IDF embedding combined with additional features
    """

    def __init__(self, n_samples=100,
                 vectorizer: str = CustomTfidf()):
        self.n_samples = n_samples
        self.vectorizer = vectorizer
        self.adding_indirect_f = _AddingFeatures()

    def fit(self, x, y=None):
        self.vectorizer.fit(x)
        self.adding_indirect_f.fit(x)
        return self

    def transform(self, x, y=None):
        tfidf_X = self.vectorizer.transform(x)
        indirect_sparse_f = self.adding_indirect_f.transform(x)
        return hstack((tfidf_X, indirect_sparse_f))
#todo : and here
    def fit_transform(self, x, y=None):
        """TFIDF part"""
        tfidf_X = self.vectorizer.fit_transform(x)  # pd.DataFrame ->
        # tfidf_X.save()  # -> tfidf.pickle
        """Adding features part"""
        indirect_sparse_f = self.adding_indirect_f.add(tfidf_X)
        """Stack"""
        return hstack((tfidf_X, indirect_sparse_f))  # -> csr_matrix


if __name__ == '__main__':
    from src.utils.utils import ReadPrepare, Split
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import SVC, LinearSVC
    from sklearn.metrics import classification_report, confusion_matrix

    path = "../../../../DB's/Toxic_database/tox_train.csv"

    """ReadPrepare & Split parts"""
    df = ReadPrepare(path, 1200).data_process()
    train_X, train_y = Split(df=df).get_train_data()
    a = CustomTfidf().fit_transform(train_X)

    """Baseline model"""
    model = LinearSVC(random_state=42, tol=1e-5)
    # -----------------------------------------

    from optuna.integration import OptunaSearchCV
    from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import cross_val_score

    svc_pipeline = make_pipeline(Preprocessor(), SVC(random_state=42, kernel='linear', degree=1))
    """init hyper-param"""
    param_distributions = {"preprocessor__??": optuna.distributions.FloatDistribution(1e-10, 1e10)}  # ,
    # param_distributions = {"svc__C": optuna.distributions.FloatDistribution(1e-10, 1e10)}  # ,
    # "svc__gamma":optuna.distributions.FloatDistribution(1e-4,1)}
    pipeline = OptunaSearchCV(svc_pipeline, param_distributions,
                              cv=StratifiedKFold(n_splits=3, shuffle=True),
                              n_trials=1, random_state=42, verbose=0, scoring=f1_score)

    """Train & predict"""
    pipeline.fit(train_X, train_y)
    pred_y = pipeline.predict(train_X)

    # -------------------------------------------
    """Fit transform"""
    pl = make_pipeline(Preprocessor(),
                       model)  # <class 'scipy.sparse._csr.csr_matrix'> ### (1071, 621) ###(0, 384)	0.14612288377660107
    pl.fit(train_X, train_y)
    pred_y = pl.predict(train_X)
    "metrics"
    print(pd.DataFrame(classification_report(y_true=train_y, y_pred=pred_y, output_dict=1,
                                             target_names=['non-toxic', 'toxic'])).transpose())
    print(confusion_matrix(y_true=train_y, y_pred=pred_y))
