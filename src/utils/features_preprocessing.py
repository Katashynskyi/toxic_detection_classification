import pandas as pd
import numpy as np
from string import punctuation
from wordcloud import STOPWORDS
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
import pickle
from src.utils.spacy_vectorizer import SpacyVectorTransformer


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
        self._tfidf = TfidfVectorizer(ngram_range=ngram_range, max_df=max_df, min_df=min_df, norm=norm,
                                      max_features=max_feat)
        self._dump = None

    def fit(self, x, y=None):
        """
        Fits TF-IDF vectorizer

        Parameters:
        ----------
        x : pandas DataFrame
            The input data to fit the TF-IDF vectorizer.
        y : ignored

        Returns:
        --------
        self
        """
        self._dump = self._tfidf.fit(x)
        return self

    def transform(self, x, y=None):
        """
         Transform into a TF-IDF sparse matrix.

         Parameters:
        ----------
        x : pandas DataFrame
            The input data to fit the TF-IDF vectorizer.
        y : ignored

        Returns:
        --------
        csr_matrix
            The sparse matrix of TF-IDF features.
        """
        self._dump = self._tfidf.transform(x)
        return self._dump

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

    Returns:
    --------
    df : csr_matrix
        TF-IDF embedding combined with additional features
    """

    def __init__(self, vectorizer_type=None, scaler=MinMaxScaler()):
        # if vectorizer_type == 'tfidf' or 'spacy':
        #     pass
        # else: raise ValueError("Invalid embedding type: choose 'tfidf' or 'spacy'")
        if vectorizer_type not in 'tfidf' or 'spacy': raise ValueError("Invalid embedding type: choose 'tfidf' or 'spacy'")
        if vectorizer_type =='tfidf':
            self._vectorizer=CustomTfidf()
        elif vectorizer_type == 'spacy':
            pass
            # self._vectorizer=???
        self.vectorizer_type= vectorizer_type
        self.scaler = scaler

    def fit(self, x, y=None):
        """
        Fits TF-IDF vectorizer

        Parameters:
        ----------
        x : pandas DataFrame
            The input data to fit the TF-IDF vectorizer.
        y : ignored

        Returns:
        --------
        self
        """
        if self.vectorizer_type == 'tfidf':
            self._vectorizer.fit(x)
        elif self.vectorizer_type == 'spacy':
            pass
        return self

    def transform(self, x, y=None):
        """
         Transform into a TF-IDF sparse matrix.

         Parameters:
        ----------
        x : pandas DataFrame
            The input data to fit the TF-IDF vectorizer.
        y : ignored

        Returns:
        --------
        csr_matrix
            The sparse matrix of TF-IDF features.
        """
        if self.vectorizer_type == 'tfidf':
            tfidf = self._vectorizer.transform(x)

            # AddingFeatures create
            additional_features = AddingFeatures().create(input_df=x)
            # print(additional_features)

            # AddingFeatures normalize
            normalize = AddingFeatures(self.scaler).normalize(additional_features)

            # AddingFeatures stack
            stack = AddingFeatures().stack(tfidf, normalize)
            return stack  # -> csr_matrix
        elif self.vectorizer_type == 'spacy':
            import spacy
            nlp = spacy.load("en_core_web_md")
            vectorizer_type = SpacyVectorTransformer(nlp=nlp)
            return vectorizer_type

    def fit_transform(self, x, y=None):
        """
        Fits & transfrorms the TF-IDF vectorizer, creates indirect features, stack them together with TF-IDF embeddings & normalize them.

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
        if self.vectorizer_type == 'tfidf':
            # CustomTfidf
            tfidf = self._vectorizer.fit_transform(x)

            # AddingFeatures create
            additional_features = AddingFeatures().create(input_df=x)
            # print(additional_features)

            # AddingFeatures normalize
            normalize = AddingFeatures().normalize(additional_features)
            # print(normalize)
            # AddingFeatures stack
            stack = AddingFeatures().stack(tfidf, normalize)
            return stack  # -> csr_matrix
        elif self.vectorizer_type == 'spacy':
            import spacy
            nlp = spacy.load("en_core_web_md")
            vectorizer_type = SpacyVectorTransformer(nlp=nlp)
            return vectorizer_type


if __name__ == '__main__':
    from src.utils.utils import ReadPrepare, Split

    # path
    path = "../../../../DB's/Toxic_database/tox_train.csv"

    # ReadPrepare test
    df = ReadPrepare(path, 1200).data_process()

    # Split test
    train_X, train_y = Split(df=df).get_train_data()

    # # CustomTfidf test
    # tfidf = CustomTfidf().fit_transform(train_X)

    # # AddingFeatures create test
    # additional_features = AddingFeatures().create(input_df=train_X)
    # print(f"additional_features: {type(additional_features)}")

    # # AddingFeatures normalize test
    # normalize = AddingFeatures().normalize(additional_features)
    # print(f"normalize: {type(normalize)}")

    # # AddingFeatures stack test
    # stack = AddingFeatures().stack(tfidf, normalize)
    # print(f"stack: {type(stack)}")

    # Preprocessor test
    # p = Preprocessor(vectorizer_type='spacy').fit_transform(train_X)
    # p = Preprocessor().fit(train_X).transform(train_X)
    # print(p.nlp) #  spacy.lang.en.English object at 0x00000264AEE66AC0
    # print(p.dim) #  300
    # print(p.__class__) #  class '__main__.SpacyVectorTransformer'