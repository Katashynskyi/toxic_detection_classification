from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd


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

    def __init__(
        self,
        max_df: float = 0.8,
        min_df: float = 10,
        ngram_range: tuple = (1, 1),
        max_feat=None,
        norm="l2",
    ):
        self._tfidf = TfidfVectorizer(
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            norm=norm,
            max_features=max_feat,
        )
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
        save_path = "../../data/tfidf.pickle"
        with open(save_path, "wb") as file:
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
        load_path = "../../data/tfidf.pickle"
        with open(load_path, "rb") as file:
            return pickle.load(file)


if __name__ == "__main__":
    from src.utils.utils import ReadPrepare, Split

    # path
    path = "../../../../DB's/Toxic_database/tox_train.csv"

    # ReadPrepare test
    df = ReadPrepare(path, 500).data_process()

    # Split test
    train_X, train_y = Split(df=df).get_train_data()

    # CustomTfidf test
    tfidf = CustomTfidf().fit_transform(train_X)
    print(tfidf)
    # print(type(tfidf))
    # print((tfidf).shape)
