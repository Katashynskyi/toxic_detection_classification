import pandas as pd
from string import punctuation
from wordcloud import STOPWORDS
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix, hstack
import pickle


# first Tfidf, thenAddingFeatures then combine.


class Tfidf:
    def __init__(self,
                 max_df: float = 0.8,
                 min_df: int or float = 10,
                 ngram_range: tuple = (1, 1)):
        self.ngram_range = ngram_range
        self.max_df = max_df
        self.min_df = min_df

        self.tfidf = TfidfVectorizer(ngram_range=ngram_range, max_df=max_df, min_df=min_df)

    def fit(self, x: pd.DataFrame):
        self.tfidf_magic = self.tfidf.fit(x)
        return self.tfidf_magic

    def transform(self, x):
        self.tfidf_magic = self.tfidf.transform(x)
        return self.tfidf_magic

    def fit_transform(self, x: pd.DataFrame):
        self.tfidf_magic = self.tfidf.fit_transform(x)
        return self.tfidf_magic

    def save(self):
        with open('data/tfidf_magic.pk', 'wb') as f:
            pickle.dump(self.tfidf_magic, f)

    def load(self):
        with open('data/tfidf_magic.pk', 'rb') as f:
            return pickle.load(f)


class AddingFeatures:
    """Adding inderect features to pd.DataFrame"""

    def __init__(self, input_df):
        self.input_df = input_df
        self.get()

    def get(self) -> pd.DataFrame:
        input_df = self.input_df.to_frame(name='comment_text')
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
        return input_df


class CompileFeatures:
    """Compile inderect features with td-idf features"""

    inderect_f = {'count_word', 'count_unique_word', 'count_letters', 'count_punctuations', 'count_words_upper',
                  'count_words_title', 'count_stopwords', 'mean_word_len', 'word_unique_percent', 'punct_percent'}

    def __init__(self, x: pd.DataFrame,
                 scaler=MinMaxScaler()):  # also add Scale, norm
        self.x = x
        self.scaler = scaler

    def fit_transform(self):
        return self.scaler.fit_transform(self.x[self.inderect_f])

    def convert_np_to_csr(self):
        return csr_matrix(self.fit_transform())

    def stack(self,Tfidf,sparce_matrix):
        return hstack((Tfidf, sparce_matrix))

    # def get(self):
    #     pass


#  train
# a=Tf()
# a.fit()
# a.transform()
# a.save()

#  inference
# b = Tfidf
# b.load()
# b.transform()

# print(f'Train TFIDF size: {tfidf_X_train.shape, type(tfidf_X_train)}')
# print(
#     f'Train Indirect features size: {sparce_scaled_X_train_only_features.shape, type(sparce_scaled_X_train_only_features)}')
# print(f'Train result of TFIDF+Indirect_features: {sparce_train.shape, type(sparce_train)}')
# print(f'\n\nTest TFIDF size: {tfidf_X_test.shape, type(tfidf_X_test)}')
# print(
#     f'Test Indirect features size: {sparce_scaled_X_test_only_features.shape, type(sparce_scaled_X_test_only_features)}')
# print(f'Test result of TFIDF+Indirect_features: {sparce_test.shape, type(sparce_test)}')
