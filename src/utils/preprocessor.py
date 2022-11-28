import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class ReadPrepare:
    def __init__(self,
                 path: str,
                 n_samples: int = 800):
        self.path = path
        self.n_samples = n_samples

    def get_file(self) -> pd.DataFrame:
        file = pd.read_csv(self.path).tail(self.n_samples)
        file.drop_duplicates(keep=False, subset=['comment_text'], inplace=True)
        file.reset_index(drop=True, inplace=True)
        return file  # <class 'pandas.core.frame.DataFrame'>& [10 rows x 8 columns]


class Split:
    def __init__(self,
                 file: pd.DataFrame,
                 test_size: float = 0.1,
                 valid_size: float = 0.1,
                 stratify_by: str = 'target_class'):
        self.file = file
        self._test_size = test_size
        self._valid_size = valid_size
        self._stratify_by = stratify_by

        self._X_train = pd.DataFrame()
        self._y_train = pd.DataFrame()
        self._X_test = pd.DataFrame()
        self._y_test = pd.DataFrame()
        self._X_valid = pd.DataFrame()
        self._y_valid = pd.DataFrame()

    def _split(self):
        df = self.file
        df = shuffle(df, random_state=0)
        df['target_class'] = (df['target'] >= 0.5).map(int)  # if more than .5 - than toxic.
        _X_temp, self._X_test, _y_temp, self._y_test = train_test_split(df['comment_text'],
                                                                        df['target_class'],
                                                                        stratify=df[self._stratify_by],
                                                                        test_size=self._test_size)
        self._X_train, self._X_valid, self._y_train, self._y_valid = train_test_split(_X_temp, _y_temp,
                                                                                      stratify=_y_temp,
                                                                                      test_size=self._valid_size)

    def get_train_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self._split()
        return self._X_train, self._y_train

    def get_test_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self._split()
        return self._X_test, self._y_test

    def get_valid_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self._split()
        return self._X_valid, self._y_valid

# path = "../../../DB's/Toxic_database/tox_train.csv"
# path = "D:/Programming/DB's/Toxic_database/tox_train.csv"
# r = ReadPrepare(path, n_samples=200)  # передаєм path
# print(r.get_file())
# s = Split(r.get_file()) # передаєм метод read_file (що підчистить датасет)
# print(s.get_train_data())  # отримати поділений датасет.
