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



class SpacyVectorTransformer(BaseEstimator):
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
import spacy
nlp = spacy.load("en_core_web_md")
# doc=SpacyVectorTransformer(nlp)
# print(doc.nlp)
import pandas as pd

a="Sorry you missed high school. Eisenhower sent troops to Vietnam after the French withdrew in 1954 and before that America was providing 50% of the cost of that war. WWI & WWII were won by Democrat Presidents and the last win a Republican had was 1865  but the did surrender in Korea and Vietnam and fail to win in desert Storm or these two wars."
b="Our oils read;  President IS taking different tactics to deal with a corrupt malignant, hipoctitical , one way press! Idiots forget what witnessed them doing during the last election process."
a=pd.Series([a,b])
docs = [nlp(text).vector for text in a]
# print(docs)
print(len(docs[0]))

