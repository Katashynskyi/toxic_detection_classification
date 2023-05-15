import argparse
import logging as logger
import os
import warnings

import lightgbm as lgb
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from mlflow import log_artifacts, log_metric, log_param
from optuna.integration import OptunaSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    make_scorer,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.utils.features_preprocessing import CustomTfidf, Preprocessor
from src.utils.spacy_vectorizer import SpacyVectorTransformer
from src.utils.utils import ReadPrepare, Split

warnings.filterwarnings("ignore")
logger.getLogger().setLevel(logger.INFO)
# logger.basicConfig(filename='log_file.log',
#                     encoding='utf-8',filemode='w')
# path = "../../DB's/Toxic_database/tox_train.csv"  # relative path
# path = "D:/Programming/DB's/Toxic_database/tox_train.csv"  # absolute path

RANDOM_STATE = 42


class ClassifierModel:
    def __init__(
        self,
        input_data,
        n_samples=800,
        n_trials=5,
        classifier_type: str = "basemodel",
        vectorizer: str = "tfidf",
        pipeline=None,
    ):
        self.input_data = input_data
        self.n_samples = n_samples
        self.n_trials = n_trials
        self.vectorizer = vectorizer
        self.classifier_type = classifier_type
        self.pipeline = pipeline
        self.x_train = self.y_train = self.x_test = self.y_test = None

    def __training_setup(self):
        # ReadPrepare
        df = ReadPrepare(self.input_data, self.n_samples).data_process()

        # Split
        self.train_X, self.train_y = Split(df=df).get_train_data()
        self.test_X, self.test_y = Split(df=df).get_test_data()

        # Model pipeline
        if self.classifier_type == "basemodel":
            svc_pipeline = make_pipeline(
                Preprocessor(vectorizer_type=self.vectorizer),
                SVC(random_state=RANDOM_STATE, kernel="linear", degree=1),
            )

            """init hyper-param"""
            param_distributions = {
                "svc__C": optuna.distributions.CategoricalDistribution(
                    [15, 20, 25, 30, 40, 50]
                )
            }  # ,
            # "svc__gamma":optuna.distributions.FloatDistribution(1e-4,1)}

            """optimization"""
            self.pipeline = OptunaSearchCV(
                svc_pipeline,
                param_distributions,
                cv=StratifiedKFold(n_splits=3, shuffle=True),
                n_trials=self.n_trials,
                random_state=42,
                verbose=0,
                scoring="f1_weighted",
            )

        elif self.classifier_type == "xgboost":
            xgb_pipeline = make_pipeline(
                Preprocessor(vectorizer_type=self.vectorizer),
                XGBClassifier(
                    learning_rate=0.05, random_state=RANDOM_STATE, missing=-999
                ),
            )

            """init hyper-param"""
            param_distributions = {
                "xgbclassifier__learning_rate": optuna.distributions.CategoricalDistribution(
                    [0.001, 0.005, 0.1]
                ),
                "xgbclassifier__max_depth": optuna.distributions.CategoricalDistribution(
                    [3, 5, 8]
                ),
                "xgbclassifier__min_child_weight": optuna.distributions.CategoricalDistribution(
                    [11, 13, 15]
                ),
                "xgbclassifier__subsample": optuna.distributions.CategoricalDistribution(
                    [0.7, 0.8]
                ),
                "xgbclassifier__n_estimators": optuna.distributions.CategoricalDistribution(
                    [500, 600, 800, 1000]
                ),
            }

            """optimization"""
            self.pipeline = OptunaSearchCV(
                xgb_pipeline,
                param_distributions,
                cv=StratifiedKFold(n_splits=3, shuffle=True),
                n_trials=self.n_trials,
                random_state=42,
                verbose=0,
                scoring=None,
            )

        elif self.classifier_type == "lgbm":
            lgbm_pipeline = make_pipeline(
                Preprocessor(vectorizer_type=self.vectorizer), lgb.LGBMClassifier()
            )
            """init hyper-param"""
            param_distributions = {}
            # param_distributions = {'num_leaves': 5,
            #           'objective': 'multiclass',
            #           'num_class': len(np.unique(self.y_train)),
            #           'learning_rate': 0.01,
            #           'max_depth': 5,
            #           'random_state': RANDOM_STATE}
            self.pipeline = OptunaSearchCV(
                lgbm_pipeline,
                param_distributions,
                cv=StratifiedKFold(n_splits=3, shuffle=True),
                n_trials=self.n_trials,
                random_state=42,
                verbose=0,
                scoring=None,
            )

    def train(self):
        """... and one method to rule them all. (c)"""
        self.__training_setup()
        train_X, train_y = self.train_X, self.train_y

        """MLFlow Config"""
        logger.info("Setting up MLFlow Config")

        mlflow.set_experiment("Toxicity_classifier_model")

        """Search for previous runs and get run_id if present"""
        # logger.info("Searching for previous runs for given model type")
        # df_runs = mlflow.search_runs(filter_string="tags.Model = '{0}'".format('XGB'))
        # df_runs = df_runs.loc[~df_runs['tags.Version'].isna(), :] if 'tags.Version' in df_runs else pd.DataFrame()
        # run_id = df_runs.loc[df_runs['tags.Version'] == run_version, 'run_id'].iloc[0]
        # run_id =3
        # load_prev = True
        # run_version = len(df_runs) + 1

        """Start the MLFlow Run and train the model"""
        with mlflow.start_run():
            """Train & predict"""
            self.pipeline.fit(train_X, train_y)
            logger.info("train is done")
            pred_y = self.pipeline.predict(train_X)
            # self.pipeline.save()

            """classification report of train set"""
            df = pd.DataFrame(
                classification_report(
                    y_true=train_y,
                    y_pred=pred_y,
                    output_dict=1,
                    target_names=["non-toxic", "toxic"],
                )
            ).transpose()

            """cross_val_score(nested_CV) of train set"""
            # kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            # score=cross_val_score(self.pipeline,train_X,train_y,cv=StratifiedKFold(n_splits=3, shuffle=True))
            # logger.info(f"cross_val_score : {score.mean()}")

            """Show train metrics"""
            logger.info(
                f"\n{pd.DataFrame(classification_report(train_y, pred_y, output_dict=1, target_names=['non-toxic', 'toxic'])).transpose()}"
            )
            logger.info(
                f"\n    Area Under the Curve score: {round(roc_auc_score(train_y, pred_y), 2)}"
            )
            logger.info(
                f"\n [true negatives  false positives]\n [false negatives  true positives] \
                        \n {confusion_matrix(train_y, pred_y)}"
            )

            # logger.info(" Logging results into file_log.log")
            logger.info("Training Complete. Logging results into MLFlow")

            """Log train metrics"""
            # Precision
            mlflow.log_metric("Precision", np.round(df.loc["toxic", "precision"], 2))

            # Recall
            mlflow.log_metric("Recall", np.round(df.loc["toxic", "recall"], 2))

            # macro_f1
            mlflow.log_metric("Macro_f1", np.round(df.loc["macro avg", "f1-score"], 2))

            # weighted_f1
            mlflow.log_metric(
                "Weighted_f1", np.round(df.loc["weighted avg", "f1-score"], 2)
            )

            # Best Score
            mlflow.log_metric("Best Score", "%.2f " % self.pipeline.best_score_)

            # AUC
            mlflow.log_metric("AUC", round(roc_auc_score(train_y, pred_y), 2))

            # Confusion matrix
            conf_matrix = confusion_matrix(train_y, pred_y)
            mlflow.log_metric("TP", conf_matrix[0][0])
            mlflow.log_metric("TN", conf_matrix[1][1])
            mlflow.log_metric("FP", conf_matrix[0][1])
            mlflow.log_metric("FN", conf_matrix[1][0])

            # df = df.reset_index()
            # df.columns = ['category', 'precision', 'recall', 'f1-score', 'support']
            # df.to_csv("toxicity_full_report.csv")
            # mlflow.log_artifact("toxicity_full_report.csv")
            # os.remove("toxicity_full_report.csv")

            """Log hyperparams"""
            # best of hyperparameter tuning
            mlflow.log_param(
                "Best Params",
                {k: round(v, 2) for k, v in self.pipeline.best_params_.items()},
            )

            # number of input comments
            mlflow.log_param("n_samples", self.n_samples)

            # vectorizer type
            mlflow.log_param("vectorizer", self.vectorizer)

            # number of trials of of hyperparameter tuning
            mlflow.log_param("n_trials", self.n_trials)

            """log model type"""
            mlflow.set_tag("Model", self.classifier_type)

            """Log(save) model"""
            # mlflow.sklearn.log_model(self.pipeline,
            #                          artifact_path='D:\Programming\Repositories\\toxic_detection_classification\data\model_log',
            #                          serialization_format='pickle')
            # mlflow.set_tag("Version", run_version)
            # logger.info("Model Trained and saved into MLFlow artifact location")

            """Test set metrics """
            # predict_y = pipeline.predict(test_X)
            # logger.info(f"\n\nTest metrics.")
            # logger.info(classification_report(test_y, pred_y, target_names=['Not_toxic', 'Toxic']))
            # logger.info(f"Area Under the Curve score: {round(roc_auc_score(test_y, pred_y), 2)}")
            # logger.info(f"\n [true negatives  false negatives]\n [true positives  false positives]")
            # logger.info(f"\n{confusion_matrix(test_y, pred_y)}")
            # logger.info("Testing completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--path", help="Data path", default="../../DB's/Toxic_database/tox_train.csv"
    )
    parser.add_argument("--n_samples", help="How many samples to pass?", default=10000)
    parser.add_argument(
        "--n_trials", help="How many trials for hyperparameter tuning?", default=10
    )
    parser.add_argument(
        "--type_of_run", help='"train" of "inference"?', default="train"
    )
    parser.add_argument(
        "--vectorizer", help='Choose "tfidf" or "spacy"', default="tfidf"
    )
    parser.add_argument(
        "--classifier_type",
        help='Choose "basemodel", "xgboost" or "lightgbm"',
        default="xgboost",
    )
    args = parser.parse_args()
    if args.type_of_run == "train":
        classifier = ClassifierModel(
            input_data=args.path,
            n_samples=args.n_samples,
            n_trials=args.n_trials,
            vectorizer=args.vectorizer,
            classifier_type=args.classifier_type,
        )
        classifier.train()

    # if args.type_of_run=='inference':
    #     with open('data/tfidf.pk', 'r') as f:
    #         tfidf=pickle.load(f)
    #     import model in pickle too
    #     classifier = ClassifierModel()  # put inference params
    #     classifier.predict()
