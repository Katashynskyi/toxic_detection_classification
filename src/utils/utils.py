import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

RANDOM_STATE = 42

class ReadPrepare:
    def __init__(self,
                 path: str,
                 n_samples: int = -1):
        self.path = path
        self.n_samples = n_samples

    def fit(self):
        return self

    def transform(self):
        return self

    def fit_transform(self, X=None, y=None) -> pd.DataFrame:
        # print('ReadPrepare fit_transform')
        if self.n_samples:
            df = pd.read_csv(self.path).tail(self.n_samples)
        else:
            df = pd.read_csv(self.path)
        df.drop_duplicates(keep=False, subset=['comment_text'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df  # <class 'pandas.core.frame.DataFrame'>


class Split:
    def __init__(self,
                 df=None,
                 test_size: float = 0.1,
                 valid_size: float = 0.1,
                 stratify_by: str = 'target_class'):
        self._df = df
        self._test_size = test_size
        self._valid_size = valid_size
        self._stratify_by = stratify_by

        self._X_train = self._y_train = pd.DataFrame()
        self._X_test = self._y_test = pd.DataFrame()
        self._X_valid = self._y_valid = pd.DataFrame()

    def fit(self):
        return self

    def transform(self):
        return self

    def fit_transform(self):
        df = self._df
        df = shuffle(df, random_state=0)
        df['target_class'] = (df['target'] >= 0.5).map(int)  # if more than .5 - than toxic.
        _X_temp, self._X_test, _y_temp, self._y_test = train_test_split(df['comment_text'],
                                                                        df['target_class'],
                                                                        stratify=df[self._stratify_by],
                                                                        test_size=self._test_size)
        self._X_train, self._X_valid, self._y_train, self._y_valid = train_test_split(_X_temp, _y_temp,
                                                                                      stratify=_y_temp,
                                                                                      test_size=self._valid_size,
                                                                                      random_state=RANDOM_STATE)
        return self

    def get_train_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.fit_transform()
        return self._X_train, self._y_train

    def get_test_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.fit_transform()
        return self._X_test, self._y_test

    def get_valid_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.fit_transform()
        return self._X_valid, self._y_valid


# path = "D:/Programming/DB's/Toxic_database/tox_train.csv"
# path = "../../../../DB's/Toxic_database/tox_train.csv"
#
# rp = ReadPrepare(path=path, n_samples=100).fit_transform()  # csv -> pd.DataFrame
# splitter = Split(df=rp).fit_transform()  # pd.DataFrame ->
# train_X, train_y = splitter.get_train_data()  # -> pd.DataFrame
# test_X, test_y = splitter.get_test_data()  # -> pd.DataFrame
# valid_X, valid_y = splitter.get_valid_data()  # -> pd.DataFrame
