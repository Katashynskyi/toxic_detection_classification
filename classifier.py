from src.utils.preprocessor import ReadPrepare, Split
from src.utils.tf_idf_vectorizer import Tfidf, AddingFeatures,CompileFeatures


path = "../../../DB's/Toxic_database/tox_train.csv"
path = "D:/Programming/DB's/Toxic_database/tox_train.csv"

"""ReadPrepare"""
r = ReadPrepare(path, n_samples=200)#shape (200, 8)
"""Split"""
s = Split(r.get_file())  # передаєм метод read_file, щоб підчистить датасет
# print(s.get_train_data())  # отримати поділений датасет. tuple of X_train & y_train

X_train, y_train = s.get_train_data()
X_test, y_test = s.get_test_data()
"""TFIDF"""
t = Tfidf()  # X_train краще передавати в ці дужки
tfidf_X_train = t.fit_transform(X_train)
tfidf_X_test = t.transform(X_test)
# print(t.save())
# print(t.load()) #  class 'scipy.sparse._csr.csr_matrix'>   (0, 49)

"""AddingFeatures"""
a=AddingFeatures(X_train)
a2=AddingFeatures(X_test)
inderect_f = {'count_word', 'count_unique_word', 'count_letters', 'count_punctuations', 'count_words_upper',
              'count_words_title', 'count_stopwords', 'mean_word_len', 'word_unique_percent', 'punct_percent'}
# print(a.get()[inderect_f]) #  [162 rows x 10 columns]
# print(a2.get()[inderect_f]) #  [20 rows x 10 columns]

"""CompileFeatures"""
c=CompileFeatures(a.get())
c1=CompileFeatures(a2.get())
# print(c.fit_transform(),type(c.fit_transform())) #  <class 'numpy.ndarray'>
# print(c.convert_np_to_csr(),type(c.convert_np_to_csr())) #  <class 'scipy.sparse._csr.csr_matrix'>

sparce_train=c.stack(tfidf_X_train,c.convert_np_to_csr()) #  numpy.ndarray to sparse matrix numpy.float64
# print(sparce_train)
sparce_test=c.stack(tfidf_X_test,c1.convert_np_to_csr()) #  numpy.ndarray to sparse matrix numpy.float64
print(sparce_test) # OK






"""tfidf_X_train = tfidf.fit_transform(X_train)
tfidf_X_test = tfidf.transform(X_test)

X_train_w_inderect_f = adding_inderect_features(X_train)
X_test_w_inderect_f = adding_inderect_features(X_test)

inderect_f = set(list(X_train_w_inderect_f.columns)) - set(list(X_train.to_frame(name='comment_text')))

scaler_1 = MinMaxScaler()  # also add Scale, norm
scaled_X_train_only_features = scaler_1.fit_transform(X_train_w_inderect_f[inderect_f])

scaled_X_test_only_features = scaler_1.fit_transform(X_test_w_inderect_f[inderect_f])


sparce_scaled_X_train_only_features = scipy.sparse.csr_matrix(scaled_X_train_only_features)  # convert np matrix to csr matrix

sparce_train = hstack((tfidf_X_train, sparce_scaled_X_train_only_features))
1416514x10 numpy.ndarray to sparse matrix numpy.float64
sparce_scaled_X_test_only_features = scipy.sparse.csr_matrix(
    scaled_X_test_only_features)  # convert np matrix to csr matrix

sparce_test = hstack((tfidf_X_test, sparce_scaled_X_test_only_features))"""