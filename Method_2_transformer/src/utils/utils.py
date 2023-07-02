import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

desired_width = 1000
pd.set_option("display.width", desired_width)
pd.set_option("display.max_columns", 100)
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

    def __init__(self, path: str, n_samples: int = -1, tox_threshold=0.5):
        self.path = path
        self.n_samples = n_samples
        self.tox_threshold = tox_threshold

    def data_process(self) -> pd.DataFrame:
        if self.n_samples:
            df = pd.read_csv(self.path).tail(self.n_samples)
        else:
            df = pd.read_csv(self.path)
        df.drop_duplicates(
            keep=False, subset=["comment_text"], inplace=True
        )  # duplicates partly left

        # remove links
        df["comment_text"] = df["comment_text"].str.replace(
            r"http\S+", "<URL>", regex=True
        )

        # Cut long comments
        df["comment_text"] = df["comment_text"].str.slice(0, 300)

        df.reset_index(drop=True, inplace=True)
        # df["labels"] = (df["target"] >= self.tox_threshold).map(int) #for float df
        label_columns = [
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        ]
        df["labels"] = df[label_columns].apply(lambda x: list(x), axis=1)
        # df['stratify'] = df['labels'].apply(lambda x: ''.join(str(e) for e in x))
        df.drop(["id"], inplace=True, axis=1)
        # df.drop(self.label_columns, inplace=True, axis=1)
        return df


class Split:
    """
    Splitting pd.DataFrame into training & testing & validation sets.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to be split into training and testing sets.
    valid_size : float, optional (default=0.2)
        The proportion of the DataFrame to use for validation.
    test_size : float, optional (default=0.2)
        The proportion of the DataFrame to use for testing.
    stratify_by : str, optional (default='threat')
        Use column name for stratification.

    Returns:
    --------
    .get_train_data(): -> tuple[pd.DataFrame, pd.DataFrame
        X_train, y_train : A separated DataFrame
    or

    .get_valid_data(): -> tuple[pd.DataFrame, pd.DataFrame]
        X_valid, y_valid : A separated DataFrame
    or

    .get_test_data(): -> tuple[pd.DataFrame, pd.DataFrame]
        X_test, y_test : A separated DataFrame
    """

    def __init__(
        self,
        df=None,
        valid_size: float = 0.2,
        test_size: float = 0.3,
        stratify_by: str = "threat",
    ):
        self._df = df
        self._valid_size = valid_size
        self._test_size = test_size
        self._stratify_by = stratify_by

        self._X_temp = self._y_temp = pd.DataFrame()
        self._X_train = self._y_train = pd.DataFrame()
        self._X_test = self._y_test = pd.DataFrame()
        self._X_valid = self._y_valid = pd.DataFrame()

    def _split(self):
        """
        Processing the input DataFrame into train and test sets.
        Use get_train_data & get_test_data methods .

        Returns:
            self
        """
        df = self._df
        df = shuffle(df, random_state=RANDOM_STATE)
        self._X_temp, self._X_test, self._y_temp, self._y_test = train_test_split(
            df["comment_text"],
            df["labels"],
            stratify=df[self._stratify_by],
            test_size=self._test_size,
            random_state=RANDOM_STATE,
        )
        self._X_train, self._X_valid, self._y_train, self._y_valid = train_test_split(
            self._X_temp,
            self._y_temp,
            # stratify=df[self._stratify_by],
            test_size=self._valid_size,
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
        return pd.concat([self._X_train, self._y_train], axis=1)

    def get_train_feature_count(self):
        self._split()
        return

    def get_valid_data(self):  # -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splitting df

        Returns:
            A tuple of X_valid & y_valid
        """
        self._split()
        return pd.concat([self._X_valid, self._y_valid], axis=1)

    def get_test_data(self):  # -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splitting df.

        Returns:
            A tuple of X_test & y_test
        """
        self._split()
        return pd.concat([self._X_test, self._y_test], axis=1)


if __name__ == "__main__":
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC

    # Path's
    # PATH = "D:/Programming/db's/toxicity_kaggle_1/train.csv" # work_PC
    # PATH2 = "D:/Programming/db's/toxicity_kaggle_1/test.csv" # work_PC
    PATH = "D:/Programming/DB's/toxic_db_for_transformert/train.csv"  # home_PC
    # PATH2 = "D:/Programming/DB's/toxic_db_for_transformert/test.csv"  # home_PC

    # path = "../../../../../DB's/Toxic_database/tox_train.csv"
    # ReadPrepare test
    rp = ReadPrepare(path=PATH, n_samples=600).data_process()  # csv -> pd.DataFrame
    # Split test
    splitter = Split(df=rp, test_size=0.3)
    train_data = splitter.get_train_data()  # -> pd.DataFrame
    test_data = splitter.get_test_data()  # -> pd.DataFrame
    print(f"train_data:\n{train_data.tail(1)}")
    # print(f"test_data:\n{test_data.tail(1)}")
