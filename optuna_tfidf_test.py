from sklearn.feature_extraction.text import TfidfVectorizer
import optuna
from optuna.trial import Trial as trial
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from src.utils.utils import ReadPrepare


class _CustomTfidf:
    """
    Creates customized TF-IDF vectorizer.

    Parameters:
    -----------
    max_df : float, optional (default=0.8)
        The maximum document frequency of a term in the corpus. Terms with a higher
        document frequency will be removed.
    min_df : float, optional (default=10)
        The minimum document frequency of a term in the corpus. Terms with a lower
        document frequency will be removed.
    ngram_range : tuple, optional (default=(1,1))
        The range of n-gram sizes to include in the TF-IDF vectorizer.

    Returns:
    --------
    tfidf in file.pickle
    """

    def __init__(self,
                 max_df: float = 0.8,
                 min_df: float = 10,
                 ngram_range: tuple = (1, 1)):
        self._tfidf = TfidfVectorizer(input=None, ngram_range=(1, 1), max_df=max_df, min_df=min_df)
        self._dump = None

    def best_tfidf_params(self):
        # ReadPrepare & Split parts
        # path = "../../../../DB's/Toxic_database/tox_train.csv"
        path = "D:/web/tox_train.csv"
        df = ReadPrepare(path, 1200).data_process()
        print(df.columns)
        # print(df["comment_text"])
        print(df["target"])

        X_train, X_test, y_train, y_test = train_test_split(df["comment_text"], df["target_class"],
                                                            test_size=0.33, random_state=42)

        # Hyperparams
        ngram_range = trial.suggest_categorical('ngram_range', [(1,1), (1,2), (2,2)])
        max_df = trial.suggest_float('max_df', 0.1, 1.0)
        min_df = trial.suggest_float('min_df', 0.0, 0.1)
        n = trial.suggest_int('n', low=1, high=3)

        # Tfidf
        vectorizer = TfidfVectorizer(input=None, ngram_range=(1, n), max_df=max_df, min_df=min_df)
        X_train_tfidf = vectorizer.fit_transform(X_train)

        # Vectorize the testing data using the same vectorizer
        X_test_tfidf = vectorizer.transform(X_test)

        # Train a linear model on the vectorized training data
        model = SVC(C=0.5,random_state=42, tol=1e-5)
        model.fit(X_train_tfidf, y_train)

        # Make predictions on the testing data
        y_pred = model.predict(X_test_tfidf)



        ### Make predictions and calculate F1 score

        y_pred = model.predict(X_test)
        score = f1_score(y_test, y_pred)
        print(score)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(best_tfidf_params, n_trials=10)
    print(study.best_trial)
