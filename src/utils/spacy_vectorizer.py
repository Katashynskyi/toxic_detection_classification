import numpy as np
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
    # TODO: додати фічі з spacy as: частину мови, частину речення, etc
    from src.utils.utils import ReadPrepare, Split
    from features_preprocessing import AddingFeatures

    # path
    path = "../../../../DB's/Toxic_database/tox_train.csv"

    # ReadPrepare test
    df = ReadPrepare(path, 500).data_process()

    # Split test
    train_X, train_y = Split(df=df).get_train_data()

    # Spacy vectorizer test
    spacy_vectorizer = SpacyVectorTransformer().fit_transform(train_X)
    print(spacy_vectorizer)
    print(spacy_vectorizer.shape)
