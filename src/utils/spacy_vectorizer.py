import numpy as np
import cupy as cp
import spacy
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

nlp = spacy.load("en_core_web_sm")


class SpacyVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, nlp=nlp):
        self.nlp = nlp

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        Doc = np.array([nlp(text).vector for text in X])
        Doc = Doc.reshape((len(X), -1))
        Doc = csr_matrix(Doc)
        return Doc


if __name__ == "__main__":
    import time
    import psutil

    # TODO: додати фічі з spacy as: частину мови, частину речення, etc
    from src.utils.utils import ReadPrepare, Split
    from features_preprocessing import AddingFeatures

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

    # Spacy vectorizer test
    spacy_vectorizer = SpacyVectorTransformer().fit_transform(train_X)
    print(spacy_vectorizer)
    print(spacy_vectorizer.shape)

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

    # Training time: 59.59 seconds Memory usage: 119.0 MB no pipe
    # Training time: 76.23 seconds Memory usage: 83.0 MB pipe
    # Training time: 105.49 seconds Memory usage: 2670.0 MB pipe witn n_jobs=-1
