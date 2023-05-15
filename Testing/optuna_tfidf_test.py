import optuna
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from src.utils.utils import ReadPrepare


class TfidfTuning:
    def best_tfidf_params(trial):
        # ReadPrepare & Split parts
        # path = "D:/web/tox_train.csv" #  work path
        path = "../../../DB's/Toxic_database/tox_train.csv"  # home path
        df = ReadPrepare(path, n_samples=10000).data_process()
        X_train, X_test, y_train, y_test = train_test_split(
            df["comment_text"],
            df["target_class"],
            test_size=0.33,
            random_state=42,
            shuffle=True,
            stratify=df["target_class"],
        )

        # Hyperparams
        # ngram_range = trial.suggest_categorical('ngram_range', [(1, 1), (1, 2), (2, 2)])
        # max_df = trial.suggest_float('max_df', 0.1, 1.0)
        C = trial.suggest_float("svc__C", 0.1, 1)
        # min_df = trial.suggest_float('min_df', 0.0, 10)
        # n = trial.suggest_int('ngram_range', low=1, high=3)

        # Tfidf
        vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=0.8, min_df=10)
        X_train_tfidf = vectorizer.fit_transform(X_train)

        # Train a linear model on the vectorized training data
        # model = SVC(C=0.2891, random_state=42, tol=1e-5)
        model = SVC(C=C, random_state=42, tol=1e-5)
        model.fit(X_train_tfidf, y_train)

        # Vectorize the testing data using the same vectorizer
        X_test_tfidf = vectorizer.transform(X_test)

        # Make predictions on the testing data
        y_pred = model.predict(X_test_tfidf)

        #  Calculate F1 score
        score = recall_score(y_true=y_test, y_pred=y_pred)
        # print(X_train_tfidf.shape)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(best_tfidf_params, n_trials=20, n_jobs=-1)
    print(study.best_trial)


# Trial 19 finished with value: 0.03666666666666667 and parameters:
# {'svc__C': 0.9842173293248563}. Best is trial 18 with value: 0.04.
