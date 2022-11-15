# ### Contents:
# * Preparing the ground
#     * Importing Libs and Datasets
#     * Data check and preprocessing
# * Feature engineering
# * Baseline model
# * More models

# ### Importing libs and datasets
import pandas as pd
import numpy as np
import os
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud, STOPWORDS
from sklearn.preprocessing import MinMaxScaler
import scipy
from scipy.sparse import hstack
from sklearn.svm import LinearSVC  # ,SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pickle

from feature_engineering import Feature_engineering

#
# feature_eng = Feature_engineering(df=pd.DataFrame({"text":"text"}))
# feature_eng.df
# feature_eng.adding_inderect_features()
# feature_eng.df
# feature_eng.adding_inderect_features2()
path = "D:/Programming/DB's/Toxic_database/tox_train.csv"
def read_file(path):
    df = pd.read_csv(path)  # 10 sec loading


    # ### Data check and preprocessing

    df.drop_duplicates(keep=False, subset=['comment_text'], inplace=True)  # Dataset duplicates are removed
    print(df.comment_text.tail(1))
    df.reset_index(drop=True,
                   inplace=True)
    # Dropping empty ID's by resetting indexation. Now the last ID is the same as the number of comments.
    print(df.comment_text.tail(1))

    return df # Dataset duplicates are removed.
              # Dropping empty ID's by resetting indexation. Now the last ID is the same as the number of comments.
class Dataset():
    def __init__(self, input_df: pd.DataFrame, split_train_test: bool = True):
        self.data_processor(input_df)
        self._X_train = pd.DataFrame()
        self._X_test = pd.DataFrame()
        self._y_train = pd.DataFrame()
        self._y_test = pd.DataFrame()
        self.split_train_test = split_train_test


df['target_class'] = (df['target'] >= 0.5).map(int)  # if more than .5 - than toxic.
X_train, X_rem, y_train, y_rem = train_test_split(df['comment_text'], df['target_class'],
                                                  stratify=df['target_class'],
                                                  test_size=0.20)
X_test, X_valid, y_test, y_valid = train_test_split(X_rem, y_rem,
                                                    stratify=y_rem,
                                                    test_size=0.50)


# * Creating Rough toxic classification based on 0.5 target threshold
# to count clean and toxic comments (class imbalance).
# * Split to train/test/validation (X,y, by='y')

# ### Feature engineering
def adding_inderect_features(df):
    df = df.to_frame(name='comment_text')
    df.loc[:, 'count_word'] = df["comment_text"].apply(lambda x: len(str(x).split()))
    df.loc[:, 'count_unique_word'] = df["comment_text"].apply(lambda x: len(set(str(x).split())))
    df.loc[:, 'count_letters'] = df["comment_text"].apply(lambda x: len(str(x)))
    df.loc[:, "count_punctuations"] = df["comment_text"].apply(
        lambda x: len([c for c in str(x) if c in string.punctuation]))
    df.loc[:, "count_words_upper"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    df.loc[:, "count_words_title"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    df.loc[:, "count_stopwords"] = df["comment_text"].apply(
        lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
    df.loc[:, "mean_word_len"] = df["comment_text"].apply(lambda x: round(np.mean([len(w) for w in str(x).split()]), 2))
    df.loc[:, 'word_unique_percent'] = df.loc[:, 'count_unique_word'] * 100 / df['count_word']
    df.loc[:, 'punct_percent'] = df.loc[:, 'count_punctuations'] * 100 / df['count_word']
    return df

tfidf = TfidfVectorizer(ngram_range=(1, 1), max_df=0.8, min_df=10)
tfidf_X_train = tfidf.fit_transform(X_train)
tfidf_X_test = tfidf.transform(X_test)

X_train_w_inderect_f = adding_inderect_features(X_train)
X_test_w_inderect_f = adding_inderect_features(X_test)

inderect_f = set(list(X_train_w_inderect_f.columns)) - set(list(X_train.to_frame(name='comment_text')))

scaler_1 = MinMaxScaler()  # also add Scale, norm
scaled_X_train_only_features = scaler_1.fit_transform(
    X_train_w_inderect_f[inderect_f])
# numpy.ndarray 10 normalized features
scaled_X_test_only_features = scaler_1.fit_transform(X_test_w_inderect_f[inderect_f])


sparce_scaled_X_train_only_features = scipy.sparse.csr_matrix(
    scaled_X_train_only_features)  # convert np matrix to csr matrix
sparce_train = hstack((tfidf_X_train, sparce_scaled_X_train_only_features))
# 1416514x10 numpy.ndarray to sparse matrix numpy.float64
sparce_scaled_X_test_only_features = scipy.sparse.csr_matrix(
    scaled_X_test_only_features)  # convert np matrix to csr matrix

sparce_test = hstack((tfidf_X_test, sparce_scaled_X_test_only_features))


print(f'Train TFIDF size: {tfidf_X_train.shape, type(tfidf_X_train)}')
print(
    f'Train Indirect features size: {sparce_scaled_X_train_only_features.shape, type(sparce_scaled_X_train_only_features)}')
print(f'Train result of TFIDF+Indirect_features: {sparce_train.shape, type(sparce_train)}')
print(f'\n\nTest TFIDF size: {tfidf_X_test.shape, type(tfidf_X_test)}')
print(
    f'Test Indirect features size: {sparce_scaled_X_test_only_features.shape, type(sparce_scaled_X_test_only_features)}')
print(f'Test result of TFIDF+Indirect_features: {sparce_test.shape, type(sparce_test)}')


def metrics(y_true, y_pred):

    target_names = ['Toxic', 'Not_toxic']
    print(classification_report(y_true, y_pred, target_names=target_names))

    roc_auc_score(y_true=y_true, y_score=y_pred, average='samples')
    print(
        f"Area Under the Curve score: {round(roc_auc_score(y_true, y_pred), 2)}\n Is this probability of Toxic or not toxic?\n or it's a probability of 1 (Toxic) class to be toxic?")

    print('\n', confusion_matrix(y_true=y_true, y_pred=y_pred))
    print('\n', np.array([['true negatives', 'false negatives'], ['true positives', 'false positives']]))
    print('\n', np.array([['correct non-toxic predict', 'wrong not-toxic prediction'],
                          ['correct toxic prediction', 'wrong toxic prediction']]))

# y_pred=model.predict(sparce_test)
# metrics(y_test,y_pred)

# save_splitted_df's
import scipy.sparse

os.makedirs('D:/Programming/Repositories/toxic_detection_classification/Model', exist_ok=True)
X_train.to_csv('D:/Programming/Repositories/toxic_detection_classification/Model/X_train.csv', index=False)
y_train.to_csv('D:/Programming/Repositories/toxic_detection_classification/Model/y_train.csv', index=False)
X_test.to_csv('D:/Programming/Repositories/toxic_detection_classification/Model/X_test.csv', index=False)
y_test.to_csv('D:/Programming/Repositories/toxic_detection_classification/Model/y_test.csv', index=False)

scipy.sparse.save_npz("sparce_train.npz", sparce_train)  # save sparce matrix
scipy.sparse.save_npz("sparce_test.npz", sparce_test)

# load from disc
X_train = pd.read_csv("X_train.csv").iloc[:, 0]
y_train = pd.read_csv("y_train.csv").iloc[:, 0]
X_test = pd.read_csv("X_test.csv").iloc[:, 0]
y_test = pd.read_csv("y_test.csv").iloc[:, 0]
sparce_train = scipy.sparse.load_npz("sparce_train.npz")
sparce_test = scipy.sparse.load_npz("sparce_test.npz")

# ### Baseline model
df_result = pd.DataFrame(columns=["model", "description", "dataset_type"])

def save_model(model, filename='model.sav'):
    """save the model to disk"""
    pickle.dump(model, open(filename, 'wb'))

# fit and save
linearSVC = LinearSVC(random_state=0, tol=1e-5)
LinSVC_fitted = linearSVC.fit(sparce_train, y_train)  # (1416514, 57204)
save_model(LinSVC_fitted, 'LinearSVC_model.sav')

# load the model from disk
linearSVC = pickle.load(open('LinearSVC_model.sav', 'rb'))
y_pred = linearSVC.predict(sparce_test)
metrics(y_test, y_pred)

linearSVC.predict(sparce_test)

if __name__==__main__:
    main()
# # * skip SVC
#
# # fit and save
# SVC = SVC(random_state=0, gamma='auto', kernel='linear')  # processed for 10 h and freezes
# # SVC_fitted=SVC.fit(sparce_train, y_train)#(1416514, 57204)
#
#
# # * DTC Done
#
# # fit and save
# DTC = DecisionTreeClassifier(random_state=0)
# DTC_fitted = DTC.fit(sparce_train, y_train)
# save_model(DTC_fitted, 'DTC_model.sav')
#
# # save the model to disk
# filename = 'DTC_model.sav'
# pickle.dump(DTC_fitted, open(filename, 'wb'))
#
# # load the model from disk
# DTC = pickle.load(open('DTC_model.sav', 'rb'))
# y_pred = DTC.predict(sparce_test)
# metrics(y_test, y_pred)
#
# set(DTC.predict_proba(sparce_test)[:, 0])
#
# # random forest, xgboost, lgbm
#
# import winsound
#
# duration = 10000  # milliseconds
# freq = 440  # Hz
# winsound.Beep(freq, duration)
#
# # ### CSV Export
# os.makedirs('D:/Programming/Repositories/toxic_detection_classification/Model', exist_ok=True)
# df_temp.to_csv('D:/Programming/Repositories/toxic_detection_classification/Model/tox_train_featurefull')
