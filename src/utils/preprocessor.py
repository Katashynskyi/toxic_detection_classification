import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from wordcloud import STOPWORDS

class Preprocessor:
    """Class which helps in preprocessing train data"""

    def __init__(self,
                 input_df: pd.DataFrame,
                 test_size: float = 0.1,
                 valid_size: float = 0.1,
                 stratify_by: str = 'target_class'):
        self.input_df = input_df
        self.test_size = test_size
        self.valid_size = valid_size
        self.stratify_by = stratify_by
        self.read_file(input_df)
        self.split(input_df)

    def read_file(self, input_df):
        """
        Reading csv,dropping comment duplicates,
        resetting index, converting to pd.DataFrame.
        :param input_df:
        :return:
        """
        self.input_df = pd.read_csv(input_df).tail(800)  # 10 sec loading
        self.input_df.drop_duplicates(keep=False,
                                      subset=['comment_text'],
                                      inplace=True)  # Dataset duplicates are removed
        self.input_df.reset_index(drop=True,
                                  inplace=True)

    def split(self, input_df: pd.DataFrame):
        """
        Method helps split the data
        :param input_df:
        :return:
        """
        self.input_df = shuffle(self.input_df, random_state=0)
        self.input_df['target_class'] = (self.input_df['target'] >= 0.5).map(int)  # if more than .5 - than toxic.
        self._X_temp, self._X_test, self._y_temp, self._y_test = train_test_split(self.input_df['comment_text'],
                                                                                  self.input_df['target_class'],
                                                                                  stratify=self.input_df[
                                                                                      self.stratify_by],
                                                                                  test_size=self.test_size)
        self._X_train, self._X_valid, self._y_train, self._y_valid = train_test_split(self._X_temp, self._y_temp,
                                                                                      stratify=self._y_temp,
                                                                                      test_size=self.valid_size)


    def adding_inderect_features(self,input_df: pd.DataFrame):
        self.input_df = self.input_df.to_frame(name='comment_text')
        self.input_df.loc[:, 'count_word'] = self.input_df["comment_text"].apply(lambda x: len(str(x).split()))
        self.input_df.loc[:, 'count_unique_word'] = self.input_df["comment_text"].apply(lambda x: len(set(str(x).split())))
        self.input_df.loc[:, 'count_letters'] = self.input_df["comment_text"].apply(lambda x: len(str(x)))
        self.input_df.loc[:, "count_punctuations"] = self.input_df["comment_text"].apply(
            lambda x: len([c for c in str(x) if c in string.punctuation]))
        self.input_df.loc[:, "count_words_upper"] = self.input_df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
        self.input_df.loc[:, "count_words_title"] = self.input_df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
        self.input_df.loc[:, "count_stopwords"] = self.input_df["comment_text"].apply(
            lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
        self.input_df.loc[:, "mean_word_len"] = self.input_df["comment_text"].apply(lambda x: round(np.mean([len(w) for w in str(x).split()]), 2))
        self.input_df.loc[:, 'word_unique_percent'] = self.input_df.loc[:, 'count_unique_word'] * 100 / self.input_df['count_word']
        self.input_df.loc[:, 'punct_percent'] = self.input_df.loc[:, 'count_punctuations'] * 100 / self.input_df['count_word']


    def get_data(self):
        return self._X_train, self._X_test, self._X_valid, self._y_train, self._y_test, self._y_valid


# path = "../../../DB's/Toxic_database/tox_train.csv"
path = "D:/Programming/DB's/Toxic_database/tox_train.csv"
p = Preprocessor(path).get_data()
print(p)
