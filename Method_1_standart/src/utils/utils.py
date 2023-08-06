import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import emoji
import re

# Preset settings
RANDOM_STATE = 42
pd.set_option("display.width", 1000)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)


class ReadPrepare:
    """
    Read and clean CSV file.

    Clean emoji, Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers, removing duplicate comments

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

    contraction_mapping = {
        "ain't": "is not",
        "aren't": "are not",
        "can't": "cannot",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "I'd": "I would",
        "I'd've": "I would have",
        "I'll": "I will",
        "I'll've": "I will have",
        "I'm": "I am",
        "I've": "I have",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "this's": "this is",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "here's": "here is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have",
        "u.s": "america",
        "e.g": "for example",
    }

    mispell_dict = {
        "colour": "color",
        "centre": "center",
        "favourite": "favorite",
        "travelling": "traveling",
        "counselling": "counseling",
        "theatre": "theater",
        "cancelled": "canceled",
        "labour": "labor",
        "organisation": "organization",
        "wwii": "world war 2",
        "citicise": "criticize",
        "youtu ": "youtube ",
        "Qoura": "Quora",
        "sallary": "salary",
        "Whta": "What",
        "narcisist": "narcissist",
        "howdo": "how do",
        "whatare": "what are",
        "howcan": "how can",
        "howmuch": "how much",
        "howmany": "how many",
        "whydo": "why do",
        "doI": "do I",
        "theBest": "the best",
        "howdoes": "how does",
        "mastrubation": "masturbation",
        "mastrubate": "masturbate",
        "mastrubating": "masturbating",
        "pennis": "penis",
        "Etherium": "Ethereum",
        "narcissit": "narcissist",
        "bigdata": "big data",
        "2k17": "2017",
        "2k18": "2018",
        "qouta": "quota",
        "exboyfriend": "ex boyfriend",
        "airhostess": "air hostess",
        "whst": "what",
        "watsapp": "whatsapp",
        "demonitisation": "demonetization",
        "demonitization": "demonetization",
        "demonetisation": "demonetization",
    }

    def __init__(self, path: str, n_samples: int = -1, tox_threshold=0.5):
        self.path = path
        self.n_samples = n_samples
        self.tox_threshold = tox_threshold

    def clean_text(self, text):
        """
        Preprocesses text for cleaning and standardization.

        Args:
            text (str): The input text to be preprocessed.

        Returns:
            str: The preprocessed text with emoji converted, links replaced, user mentions removed,
                 numbers removed, converted to lowercase, non-alphanumeric characters removed except "?", "!", ",", and "'",
                 and extra spaces removed.
        """
        # Convert emoji to their textual representations
        text = emoji.demojize(text)

        # Replace URLs with a placeholder
        text = re.sub(r"http\S+", "<URL>", text)

        # Remove user mentions
        text = re.sub(r"@\w+", "", text)

        # Remove numbers
        text = re.sub(r"\d+", "", text)

        # Convert text to lowercase
        text = text.lower()

        # Remove non-alphanumeric characters except "?", "!", ",", and "'"
        text = re.sub(r"[^a-zA-Z\s?!,\']", "", text)

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def clean_contractions_n_mispell(self, text, mapping):
        """Clean contraction using contraction mapping"""
        specials = ["’", "‘", "´", "`"]
        for s in specials:
            text = text.replace(s, "'")
        for word in mapping.keys():
            if "" + word + "" in text:
                text = text.replace("" + word + "", "" + mapping[word] + "")
        text = re.sub(r"([?.!,¿])", r" \1 ", text)
        text = re.sub(r'[" "]+', " ", text)
        return text

    def data_process(self) -> pd.DataFrame:
        if self.n_samples:
            df = pd.read_csv(self.path, chunksize=self.n_samples).get_chunk(
                size=self.n_samples
            )
        else:
            df = pd.read_csv(self.path)
        # Apply preprocessing to the "comment_text" column
        df["comment_text"] = df["comment_text"].apply(
            lambda text: self.clean_text(text)
        )

        # substitute contractions
        df["comment_text"] = df["comment_text"].apply(
            lambda text: self.clean_contractions_n_mispell(
                text, self.contraction_mapping
            )
        )
        # # substitute mispell
        df["comment_text"] = df["comment_text"].apply(
            lambda text: self.clean_contractions_n_mispell(text, self.mispell_dict)
        )

        df.drop_duplicates(
            keep=False, subset=["comment_text"], inplace=True
        )  # duplicates partly left

        # Cut long comments
        df["comment_text"] = df["comment_text"].str.slice(0, 300)
        df.reset_index(drop=True, inplace=True)
        df["target_class"] = (df["target"] >= self.tox_threshold).map(int)
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
    # Path
    path = "../../../../../DB's/Toxic_database/tox_train.csv"
    # ReadPrepare test
    rp = ReadPrepare(path=path, n_samples=10000).data_process()  # csv -> pd.DataFrame
    print(rp.tail(3))
    # Split test
    # splitter = Split(df=rp)
    # train_X, train_y = splitter.get_train_data()  # -> pd.DataFrame
    # test_X, test_y = splitter.get_test_data()  # -> pd.DataFrame
    # print(f"train_X:\n{train_X.tail(1)}")
    # print(f"train_y:\n{train_y.tail(1)}")
    # print(f"test_X:\n{test_X.tail(1)}")
    # print(f"test_y:\n{test_y.tail(1)}")
