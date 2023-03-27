import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

RANDOM_STATE = 42


class ReadPrepare:
    """
    A class for reading in a CSV file, preprocessing it by removing duplicate
    comments, and returning a Pandas DataFrame with the remaining comments.

    Parameters:
    -----------
    path : str
        The path to the CSV file to be read.
    n_samples : int, optional (default=-1)
        The number of rows to read in from the CSV file. If set to -1 (default),
        all rows are read in.

    Returns:
    --------
    df : pandas DataFrame
        Contains non-duplicated comments from the CSV file.
    """

    def __init__(self,
                 path: str,
                 n_samples: int = -1):
        self.path = path
        self.n_samples = n_samples

    def fit(self):
        return self

    def transform(self):
        return self

    def fit_transform(self, x=None, y=None) -> pd.DataFrame:
        if self.n_samples:
            df = pd.read_csv(self.path).tail(self.n_samples)
        else:
            df = pd.read_csv(self.path)
        df.drop_duplicates(keep=False, subset=['comment_text'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df


class Split:
    """
    A class for splitting a Pandas DataFrame into training & testing sets.

    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to be split into training and testing sets.
    test_size : float, optional (default=0.1)
        The proportion of the DataFrame to use for testing. Must be between 0 and 1.
    stratify_by : str, optional (default='target_class')
        The column of the DataFrame to use for stratified sampling. Must be a column
        of binary labels (0 or 1).

    Returns:
    --------
    .get_train_data(): -> tuple[pd.DataFrame, pd.DataFrame
        X_train, y_train : A splitted DataFrame
    .get_test_data(): -> tuple[pd.DataFrame, pd.DataFrame]
        X_test, y_test : A splitted DataFrame
    """

    def __init__(self,
                 df=None,
                 test_size: float = 0.1,
                 stratify_by: str = 'target_class'):
        self._df = df
        self._test_size = test_size
        self._stratify_by = stratify_by

        self._X_train = self._y_train = pd.DataFrame()
        self._X_test = self._y_test = pd.DataFrame()
        self._X_valid = self._y_valid = pd.DataFrame()

    def fit(self):
        """Does nothing, use get_train_data & get_test_data instead."""
        return self

    def transform(self):
        """Does nothing, use get_train_data & get_test_data instead."""
        return self

    def fit_transform(self):
        """
        Processing the input DataFrame into train and test sets.
        But better use get_train_data & get_test_data instead.

        Returns:
        self
        """
        df = self._df
        df = shuffle(df, random_state=RANDOM_STATE)
        df['target_class'] = (df['target'] >= 0.5).map(int)  # if more than .5 - than toxic.
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(df['comment_text'],
                                                                                    df['target_class'],
                                                                                    stratify=df[self._stratify_by],
                                                                                    test_size=self._test_size,
                                                                                    random_state=RANDOM_STATE)
        return self

    def get_train_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splitting df.

        Returns:
            A tuple of X_train & y_train
        """
        self.fit_transform()
        return self._X_train, self._y_train

    def get_test_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splitting df.

        Returns:
            A tuple of X_test & y_test
        """
        self.fit_transform()
        return self._X_test, self._y_test

    # def get_valid_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
    #     """
    #     Splitting df.
    #
    #     Returns:
    #         A tuple of X_validation & y_validation
    #     """
    #     self.fit_transform()
    #     return self._X_valid, self._y_valid


if __name__ == '__main__':
    # path = "D:/Programming/DB's/Toxic_database/tox_train.csv"
    path = "../../../../DB's/Toxic_database/tox_train.csv"

    rp = ReadPrepare(path=path, n_samples=500).fit_transform()  # csv -> pd.DataFrame
    splitter = Split(df=rp).fit_transform()  # pd.DataFrame ->
    train_X, train_y = splitter.get_train_data()  # -> pd.DataFrame
    test_X, test_y = splitter.get_test_data()  # -> pd.DataFrame
    # valid_X, valid_y = splitter.get_valid_data()  # -> pd.DataFrame
    print(f"train_X {train_X.tail()}")
    print(f"train_y {train_y.tail()}")
    print(f"test_X {test_X.tail()}")
    print(f"test_y {test_y.tail()}")
