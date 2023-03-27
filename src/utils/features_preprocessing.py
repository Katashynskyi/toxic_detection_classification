import pandas as pd
from string import punctuation
from wordcloud import STOPWORDS
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix, hstack
import pickle
from utils import ReadPrepare, Split
from sklearn.svm import LinearSVC

class Tfidf:
    """ Creating tf-idf vectorizer
    :return: csr_matrix"""

    def __init__(self,
                 max_df: float = 0.8,
                 min_df: float = 10,
                 ngram_range: tuple = (1, 1)):
        self._tfidf = TfidfVectorizer(ngram_range=ngram_range, max_df=max_df, min_df=min_df)
        self._dump = None

    def fit(self, x: pd.DataFrame):
        self._tfidf.fit(x)
        return self

    def transform(self, x: pd.DataFrame, y=None):
        self._dump = self._tfidf.transform(x)
        return self._dump

    def fit_transform(self, x: pd.DataFrame, y=None):
        self._dump = self._tfidf.fit_transform(x)
        # with open("../../data/tfidf.pickle", 'wb') as file:
        #     pickle.dump(self.tfidf, file)
        return self._dump

    # TODO: change return?
    def save(self):
        with open("../../data/tfidf.pickle", 'wb') as file:
            pickle.dump(self._dump, file)

    def load(self):
        with open("../../data/tfidf.pickle", 'rb') as file:
            return pickle.load(file)


class AddingFeatures:
    """Adding indirect features to "comment_text" column
    and compile it with tf-idf features
    :return: csr_matrix """
    indirect_f_list = ['count_word', 'count_unique_word', 'count_letters', 'count_punctuations', 'count_words_upper',
                       'count_words_title', 'count_stopwords', 'mean_word_len', 'word_unique_percent', 'punct_percent']

    def __init__(self,
                 scaler=MinMaxScaler(),
                 indirect_f: list = indirect_f_list
                 ):
        self._scaler = scaler
        self._indirect_f = indirect_f

    def add(self, input_df) -> pd.DataFrame:  # 'DataFrame' object has no attribute 'to_frame'!!!
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
        return input_df[self._indirect_f]  # without "comment_text"

    def fit(self, x):
        new_features = self.add(x)
        self._scaler.fit(new_features)
        return self

    def transform(self, x, y=None):
        new_features = self.add(x)
        scaled_x = self._scaler.transform(new_features)
        return csr_matrix(scaled_x)

    def fit_transform(self, x, y=None):
        new_features = self.add(x)
        scaled_x = self._scaler.fit_transform(new_features)
        return csr_matrix(scaled_x)

    def stack(self, tfidf, indirect_features):
        return hstack((tfidf, indirect_features))


path = "../../../../DB's/Toxic_database/tox_train.csv"

# """ReadPrepare & Split parts"""
# # rp = ReadPrepare(path=path, n_samples=80).fit_transform()  # csv -> pd.DataFrame
# # splitter = Split(df=rp).fit_transform()  # pd.DataFrame ->
# # train_X, train_y = splitter.get_train_data()  # -> pd.DataFrame
# # test_X, test_y = splitter.get_test_data()  # -> pd.DataFrame
# # valid_X, valid_y = splitter.get_valid_data()  # -> pd.DataFrame
# """TFIDF part"""
# # tfidf = Tfidf()
# # tfidf.fit_transform(train_X)  # pd.DataFrame ->
# # tfidf.save()  # -> tfidf.pickle
# # print(Tfidf().load())  # -> csr_matrix .shape (64, 33)
# """Adding features part"""
# # adding_features = AddingFeatures()
# # print(adding_features.stack(Tfidf().load(),
# # adding_features.fit_transform(train_X)))  # -> csr_matrix


class Preprocessor:  # massive huge out of rules class
    def __init__(self, n_samples=80,
                 vectorizer=Tfidf(),
                 adder_indirect_features=AddingFeatures()):
        self.n_samples = n_samples
        self.vec = vectorizer
        self.adding_indirect_f = adder_indirect_features

    # def fit(self, x, y):
    #     self.vec.fit(x)
    #     self.adding_indirect_f.fit(x)
    #     return self
    #
    # def transform(self, x, y):
    #     tfidf_X_ = self.vec.transform(x)
    #     inderect_sparse_f = self.adding_indirect_f.transform(x)
    #     return hstack((tfidf_X_, inderect_sparse_f))

    def fit_transform(self, x, y=None):
        """ReadPrepare & Split parts"""
        #rp = ReadPrepare(path=x, n_samples=self.n_samples).fit_transform()  # csv -> pd.DataFrame
        #splitter = Split(df=rp).fit_transform()  # pd.DataFrame ->
        #train_X, train_y = splitter.get_train_data()  # -> pd.DataFrame
        # test_X, test_y = splitter.get_test_data()  # -> pd.DataFrame
        # valid_X, valid_y = splitter.get_valid_data()  # -> pd.DataFrame
        """TFIDF part"""
        tfidf_X_ = self.vec  # pd.DataFrame ->
        tfidf_X_.fit_transform(x)  # pd.DataFrame ->
        #tfidf_X_.save()  # -> tfidf.pickle
        """Adding features part"""
        indirect_sparse_f = self.adding_indirect_f.fit_transform(train_X)
        return hstack((tfidf_X_.load(), indirect_sparse_f))  # -> csr_matrix

"""RUN"""
rp = ReadPrepare(path=path).fit_transform()  # csv -> pd.DataFrame
splitter = Split(df=rp).fit_transform()  # pd.DataFrame ->
train_X, train_y = splitter.get_train_data()  # -> pd.DataFrame
# test_X, test_y = splitter.get_test_data()  # -> pd.DataFrame
# valid_X, valid_y = splitter.get_valid_data()  # -> pd.DataFrame
print(Preprocessor().fit_transform(train_X))




"""ReadPrepare & Split parts"""
rp = ReadPrepare(path=x, n_samples=80).fit_transform()  # csv -> pd.DataFrame
splitter = Split(df=rp).fit_transform()  # pd.DataFrame ->
train_X, train_y = splitter.get_train_data()  # -> pd.DataFrame

"""Baseline model"""
class BaseModel:
    def fit(self,x,y):
        pass
    def predict(self):
        pass


"""fit"""
linearSVC = LinearSVC(random_state=42, tol=1e-5)
LinSVC_fitted = linearSVC.fit(sparce_train, train_y)  # (1416514, 57204)

y_pred = linearSVC.predict(sparce_test)
metrics(y_test, y_pred)

linearSVC.predict(sparce_test)



#  train
# a=Tf()
# a.fit()
# a.transform()
# a.save()

#  inference
# b = Tfidf
# b.load()
# b.transform()


from sklearn.pipeline import make_pipeline

# pl=make_pipeline(Preprocessor(),BaseModel())
# pl.fit_transform(path)
