from sklearn.base import BaseEstimator, TransformerMixin

# class SpacyVectorTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, nlp):
#         self.nlp = nlp
#         self.dim = 300
#
#     def fit(self, X, y):
#         return self
#
#     def transform(self, X):
#         return [self.nlp(text).vector for text in X]


class SpacyVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, nlp, output_dim=300):
        self.nlp = nlp
        self.output_dim = output_dim

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        docs = [self.nlp(text) for text in X]
        outputs = [doc.cats.values() for doc in docs]
        outputs = [list(x) + [0] * (self.output_dim - len(x)) for x in outputs]
        return outputs