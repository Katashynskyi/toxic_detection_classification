import numpy as np
import spacy
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin


class SpacyVectorTransformer(BaseEstimator, TransformerMixin):
    """
    Creates customized spacy vectorizer.

    Returns:
    --------
    csr_matrix based on spacy "en_core_web_sm"
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        nlp = spacy.load(
            "en_core_web_sm",
            disable=["tagger", "parser", "attribute_ruler", "lemmatizer", "ner"],
        )  # tok2vec & senter staying
        docs = nlp.pipe(texts=X, batch_size=2000)
        vectors = np.vstack([doc.vector for doc in docs])
        return csr_matrix(vectors)


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
    df = ReadPrepare(path, 100000).data_process()

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

    # 10k Training time: 16.56 seconds Memory usage: 146.0 MB disable=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer']
    # 10k Training time: 27.83 seconds Memory usage: 107.0 MB disabled nothing
    # 100k Training time: 61.62 seconds Memory usage: 410.0 MB
    # 100k Training time: 65.5 seconds Memory usage: 1009.0 MB n_process=2
    # 100k Training time: 90.91 seconds Memory usage: 213.0 MB n_process=-1
    # 100k Training time: 59.9 seconds Memory usage: 182.0 MB n_process=None, batch_size=1000
    # 100k Training time: 59.9 seconds Memory usage: 182.0 MB n_process=None, batch_size=2000
    # 100k Training time: 67.32 seconds Memory usage: 383.0 MB n_process=None, batch_size=200
    # 100k Training time: 61.2 seconds Memory usage: 436.0 MB n_process=None, batch_size=50
    # 100k Training time: 87.93 seconds Memory usage: 298.0 MB n_process=-1, batch_size=50
    # 100k Training time: 61.02 seconds Memory usage: 261.0 MB n_process=1, batch_size=50
    # 100k Training time: 67.29 seconds Memory usage: 1419.0 MB n_process=2, batch_size=50
    # 100k Training time: 61.72 seconds Memory usage: 255.0 MB n_process=1, batch_size=1000
    # 100k Training time: 61.63 seconds Memory usage: 315.0 MB n_process=1, batch_size=2000
    # 100k Training time: 62.72 seconds Memory usage: 227.0 MB n_process=1, batch_size=None
    # 100k Training time: 61.25 seconds Memory usage: 322.0 MB n_process=None, batch_size=50
    # 100k Training time: 61.38 seconds Memory usage: 119.0 MB n_process=None, batch_size=2000

    # 100k 300vect Training time: 71.21 seconds Memory usage: 113.0 MB n_process=None, batch_size=2000
    # 100k 300vect Training time: 80.14 seconds Memory usage: 701.0 MB n_process=2, batch_size=2000
