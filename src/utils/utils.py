import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

RANDOM_STATE = 42


class ReadPrepare:
    """
    Read CSV file, removing duplicate comments.

    Parameters:
    -----------
    path : str
        The path to the CSV file to be read.
    n_samples : int, optional (default=-1)
        The number of rows to read in from the CSV file.
        If set to -1 (default) - all rows are read in.

    Returns:
    --------
    df : pd.DataFrame
        Contains non-duplicated (almost) comments from the CSV file.
    """

    def __init__(self, path: str, n_samples: int = -1):
        self.path = path
        self.n_samples = n_samples

    def data_process(self) -> pd.DataFrame:
        if self.n_samples:
            df = pd.read_csv(self.path).tail(self.n_samples)
        else:
            df = pd.read_csv(self.path)
        df.drop_duplicates(
            keep=False, subset=["comment_text"], inplace=True
        )  # частково дублікати лишились

        # Cut long comments
        df["comment_text"] = df["comment_text"].str.slice(0, 100)

        df.reset_index(drop=True, inplace=True)
        df["target_class"] = (df["target"] >= 0.5).map(
            int
        )  # if more than .5 - than toxic.
        return df


class Split:
    """
    Splitting pd.DataFrame into training & testing sets.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to be split into training and testing sets.
    test_size : float, optional (default=0.1)
        The proportion of the DataFrame to use for testing.
    stratify_by : str, optional (default='target_class')
        Use column name for stratification.

    Returns:
    --------
    .get_train_data(): -> tuple[pd.DataFrame, pd.DataFrame
        X_train, y_train : A separated DataFrame
    or

    .get_test_data(): -> tuple[pd.DataFrame, pd.DataFrame]
        X_test, y_test : A separated DataFrame
    """

    def __init__(
        self, df=None, test_size: float = 0.1, stratify_by: str = "target_class"
    ):
        self._df = df
        self._test_size = test_size
        self._stratify_by = stratify_by

        self._X_train = self._y_train = pd.DataFrame()
        self._X_test = self._y_test = pd.DataFrame()

    def _split(self):
        """
        Processing the input DataFrame into train and test sets.
        Use get_train_data & get_test_data methods .

        Returns:
            self
        """
        df = self._df
        df = shuffle(df, random_state=RANDOM_STATE)
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            df["comment_text"],
            df["target_class"],
            stratify=df[self._stratify_by],
            test_size=self._test_size,
            random_state=RANDOM_STATE,
        )
        return self

    def get_train_data(self):  # -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splitting df.

        Returns:
            A tuple of X_train & y_train
        """
        self._split()
        # return self._X_train, self._y_train
        return self._X_train, self._y_train

    def get_test_data(self):  # -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splitting df.

        Returns:
            A tuple of X_test & y_test
        """
        self._split()
        return self._X_test, self._y_test


if __name__ == "__main__":
    from features_preprocessing import Preprocessor
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC

    # Path
    path = "../../../../DB's/Toxic_database/tox_train.csv"
    # ReadPrepare test
    rp = ReadPrepare(path=path, n_samples=-1).data_process()  # csv -> pd.DataFrame
    # Split test
    splitter = Split(df=rp)
    train_X, train_y = splitter.get_train_data()  # -> pd.DataFrame
    test_X, test_y = splitter.get_test_data()  # -> pd.DataFrame
    print(f"train_X:\n{train_X.tail(1)}")
    print(f"train_y:\n{train_y.tail(1)}")
    print(f"test_X:\n{test_X.tail(1)}")
    print(f"test_y:\n{test_y.tail(1)}")
