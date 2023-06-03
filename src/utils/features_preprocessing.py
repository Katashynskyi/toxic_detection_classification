from sklearn.preprocessing import MinMaxScaler
from sklearn.base import TransformerMixin
from src.utils.spacy_vectorizer import SpacyVectorTransformer
from src.utils.custom_tf_idf_vectorizer import CustomTfidf
from src.utils.adding_features import AddingFeatures


#  meta-class hardcoded
class Preprocessor(TransformerMixin):
    """
    Hardcoded meta-class of TFIDF or Spacy features
    Preprocessor to transform raw text data into a sparse matrix of features.

    Returns:
    --------
    df : csr_matrix
        TF-IDF embedding combined with additional features

        or

        Spacy sentence embeddings with additional feature
    """

    def __init__(self, vectorizer_type="tfidf", scaler=MinMaxScaler()):
        vectorizers = {
            "tfidf": CustomTfidf(),
            "spacy": SpacyVectorTransformer(),
        }
        if vectorizer_type not in {"tfidf", "spacy"}:
            raise ValueError("Invalid embedding type: choose 'tfidf' or 'spacy'")
        self._vectorizer = vectorizers.get(vectorizer_type, None)
        self._scaler = scaler

    def fit(self, x, y=None):
        """
        Fits  TF-IDF or Spacy vectorizer

        Parameters:
        ----------
        x : pandas DataFrame
            The input data to fit the TF-IDF vectorizer.
        y : ignored

        Returns:
        --------
        self
        """
        self._vectorizer.fit(x)
        return self

    def transform(self, x, y=None):
        """
         Transform into a compressed sparse matrix using TF-IDF or Spacy embeddings.

         Parameters:
        ----------
        x : pandas DataFrame
            The input data to fit the vectorizer.
        y : ignored

        Returns:
        --------
        csr_matrix
            The sparse matrix of features.
        """
        vectorizer = self._vectorizer.transform(x)

        # AddingFeatures create
        additional_features = AddingFeatures().create(input_df=x)

        # AddingFeatures normalize
        normalize = AddingFeatures(self._scaler).normalize(additional_features)

        # AddingFeatures stack
        stack = AddingFeatures().stack(vectorizer, normalize)
        return stack  # -> csr_matrix


if __name__ == "__main__":
    import time
    import psutil
    from src.utils.utils import ReadPrepare, Split

    # Start time
    start_time = time.time()

    # Track initial memory usage
    initial_memory = psutil.virtual_memory().used

    # path
    path = "../../../../DB's/Toxic_database/tox_train.csv"

    # ReadPrepare test
    df = ReadPrepare(path, 10000).data_process()

    # Split test
    train_X, train_y = Split(df=df).get_train_data()

    # p = Preprocessor(vectorizer_type='spacy').fit(train_X).transform(train_X)
    p = Preprocessor(vectorizer_type="spacy").fit_transform(train_X)
    print(p)

    # Track final memory usage
    final_memory = psutil.virtual_memory().used

    # End time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = round((end_time - start_time), 2)
    memory_usage = final_memory - initial_memory

    # Print the elapsed time
    print(
        "Training time:",
        elapsed_time,
        "seconds",
        "Memory usage:",
        abs(round((memory_usage / 1000000), 0)),
        "MB",
    )
