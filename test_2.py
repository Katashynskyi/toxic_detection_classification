import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.utils import ReadPrepare,Split

path="D:\Programming\DB's\Toxic_database\\tox_train.csv"
def objective(trial):
    df=ReadPrepare(path,5000).data_process()
    X_train, X_test, y_train, y_test = train_test_split(df['comment_text'],df['target_class'],test_size=.33,stratify=df['target_class'],random_state=42)
    #  TFIDF vectorizer
    vectorizer=TfidfVectorizer()
    X_train_tfidf=vectorizer.fit_transform(X_train)
    X_test_tfidf=vectorizer.transform(X_test)

    #model
    # C = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
    model=SVC(C=0.2891566265060241,kernel='rbf',gamma='scale',class_weight='balanced')
    model.fit(X_train_tfidf,y_train)
    y_pred=model.predict(X_test_tfidf)
    # print(classification_report(y_test,y_pred))
    return f1_score(y_test,y_pred)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    print(study.best_trial)