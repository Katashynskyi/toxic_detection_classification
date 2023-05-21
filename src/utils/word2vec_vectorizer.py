import time
import psutil
import numpy as np
from gensim.models import Word2Vec
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin


# Load the word2vec model
# word2vec_model = Word2Vec.load("path_to_word2vec_model")  # Replace "path_to_word2vec_model" with the actual path to your word2vec model


class Word2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, word2vec_model=None):
        self.word2vec_model = word2vec_model

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        vectors = []
        for text in X:
            words = (
                text.split()
            )  # Split the text into words (assuming space-separated words)
            vector = np.mean(
                [
                    self.word2vec_model.wv[word]
                    for word in words
                    if word in self.word2vec_model.wv
                ],
                axis=0,
            )
            vectors.append(vector)

        Doc = np.vstack(vectors)
        Doc = csr_matrix(Doc)
        return Doc


if __name__ == "__main__":
    # TODO: додати фічі з spacy as: частину мови, частину речення, etc
    from src.utils.utils import ReadPrepare, Split
    from cupy.cuda import memory

    # path
    path = "../../../../DB's/Toxic_database/tox_train.csv"

    # ReadPrepare test
    df = ReadPrepare(path, 100000).data_process()

    # Split test
    train_X, train_y = Split(df=df).get_train_data()

    # Start time
    start_time = time.time()

    # Track initial memory usage
    initial_memory = psutil.virtual_memory().used

    train_X = train_X.tolist()
    print(type(train_X))
    for i in train_X:
        print(i)
    # print(train_X)
    # TODO: train model
    w2v_model = Word2Vec(
        min_count=20,
        window=2,
        vector_size=300,
        sample=6e-5,
        alpha=0.03,
        min_alpha=0.0007,
        negative=20,
        workers=-1,
    )

    # w2v_model.build_vocab(train_X, progress_per=10000)
    #
    # w2v_model.train(train_X, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

    # w2v_model.init_sims(replace=True)
    # print(w2v_model.wv.vectors)
    # print(w2v_model.wv.vectors.shape)
    # print(len(w2v_model.wv.vectors))

    # # Spacy vectorizer test
    # spacy_vectorizer = Word2VecTransformer().fit_transform(train_X)
    # print(spacy_vectorizer)
    # print(spacy_vectorizer.shape)

    # Track final memory usage
    final_memory = psutil.virtual_memory().used

    # End time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = round((end_time - start_time), 2)

    # Calculate the memory usage during training
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
