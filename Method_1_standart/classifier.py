import argparse
import logging as logger
import os
import warnings

import lightgbm as lgb
import mlflow
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
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.utils.features_preprocessing import Preprocessor
from src.utils.utils import ReadPrepare, Split

warnings.filterwarnings("ignore")
logger.getLogger().setLevel(logger.INFO)

RANDOM_STATE = 42


class ClassifierModel:
    def __init__(
        self,
        input_data,
        n_samples=800,
        tox_threshold: float = 0.5,
        n_trials=5,
        classifier_type: str = "logreg",
        vectorizer: str = "tfidf",
        pipeline=None,
        save_model=False,
    ):
        self.input_data = input_data
        self.n_samples = n_samples
        self.tox_threshold = tox_threshold
        self.n_trials = n_trials
        self.vectorizer = vectorizer
        self.classifier_type = classifier_type
        self.pipeline = pipeline
        self.x_train = self.y_train = self.x_test = self.y_test = None
        self.save_model = save_model
        self.skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    def __training_setup(self):
        # ReadPrepare
        df = ReadPrepare(
            self.input_data, self.n_samples, self.tox_threshold
        ).data_process()

        # Split
        self.train_X, self.train_y = Split(df=df).get_train_data()
        self.test_X, self.test_y = Split(df=df).get_test_data()

        # Model pipeline
        if self.classifier_type == "logreg":
            logreg_pipeline = make_pipeline(
                Preprocessor(vectorizer_type=self.vectorizer),
                LogisticRegression(
                    random_state=RANDOM_STATE, class_weight={0: 10, 1: 90}
                ),
            )

            """init hyper-param"""
            param_distributions = {
                "logisticregression__C": optuna.distributions.CategoricalDistribution(
                    [1, 10, 50, 100, 200, 500]
                ),
            }
            # "svc__gamma":optuna.distributions.FloatDistribution(1e-4,1)}

            """optimization"""
            self.pipeline = OptunaSearchCV(
                logreg_pipeline,
                param_distributions,
                cv=self.skf,
                n_trials=self.n_trials,
                random_state=RANDOM_STATE,
                verbose=0,
                scoring="recall",
            )

        elif self.classifier_type == "xgboost":
            xgb_pipeline = make_pipeline(
                Preprocessor(vectorizer_type=self.vectorizer),
                XGBClassifier(random_state=RANDOM_STATE),
            )

            """init hyper-param"""
            param_distributions = {
                "xgbclassifier__learning_rate": optuna.distributions.CategoricalDistribution(
                    [0.001, 0.005, 0.01, 0.05, 0.1]
                ),
                "xgbclassifier__max_depth": optuna.distributions.CategoricalDistribution(
                    [10, 8, 5]
                ),
                "xgbclassifier__min_child_weight": optuna.distributions.CategoricalDistribution(
                    [15, 13, 11]
                ),
                # imbalance param: subsample: 0-1
                "xgbclassifier__subsample": optuna.distributions.CategoricalDistribution(
                    [0.6, 0.7]
                ),
                "xgbclassifier__n_estimators": optuna.distributions.CategoricalDistribution(
                    [500, 600, 800, 1000]
                ),
                # imbalance param: scale_pos_weight :  sum(negative instances) / sum(positive instances)
                "xgbclassifier__scale_pos_weight": optuna.distributions.CategoricalDistribution(
                    [9.85]
                ),
                # imbalance param: max_delta_step : 1-10
                "xgbclassifier__max_delta_step": optuna.distributions.CategoricalDistribution(
                    [10]
                ),
            }

            """optimization"""
            self.pipeline = OptunaSearchCV(
                xgb_pipeline,
                param_distributions,
                cv=self.skf,
                n_trials=self.n_trials,
                random_state=RANDOM_STATE,
                verbose=0,
                scoring="roc_auc",
            )

        elif self.classifier_type == "lgbm":
            lgbm_pipeline = make_pipeline(
                Preprocessor(vectorizer_type=self.vectorizer),
                lgb.LGBMClassifier(
                    random_state=RANDOM_STATE,
                ),
            )
            """init hyper-param"""
            param_distributions = {
                # "lgbmclassifier__learning_rate": optuna.distributions.CategoricalDistribution(
                #     [0.001, 0.005, 0.01, 0.05, 0.1] #default 0.1 is best
                # ),
                # "lgbmclassifier__max_depth": optuna.distributions.CategoricalDistribution(
                #     [10, 8, 5] # spoils a lot
                # ),
                "lgbmclassifier__num_leaves": optuna.distributions.CategoricalDistribution(
                    [23]
                ),
                # "lgbmclassifier__min_child_weight": optuna.distributions.CategoricalDistribution(
                #     [15,13,11] # spoils a lot
                # ),
                "lgbmclassifier__scale_pos_weight": optuna.distributions.CategoricalDistribution(
                    [9.8]
                ),
                "lgbmclassifier__n_estimators": optuna.distributions.CategoricalDistribution(
                    [500, 600, 800, 1000]
                ),
                # "lgbmclassifier__"
            }
            # param_distributions = {'num_leaves': 5,
            #           'objective': 'multiclass',
            #           'num_class': len(np.unique(self.y_train)),
            #           'learning_rate': 0.01,
            #           'max_depth': 5,
            #           'random_state': RANDOM_STATE}
            self.pipeline = OptunaSearchCV(
                lgbm_pipeline,
                param_distributions,
                cv=self.skf,
                n_trials=self.n_trials,
                random_state=RANDOM_STATE,
                verbose=0,
                scoring="recall",
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

        """Train model and save train metrics to mlflow"""
        with mlflow.start_run():
            mlflow.set_tag(
                "mlflow.runName", f"train_{mlflow.active_run().info.run_name}"
            )
            """Train & predict"""
            self.pipeline.fit(train_X, train_y)
            logger.info("train is done")
            logger.info("↓↓↓ TRAIN METRICS ↓↓↓")

            """Get mean train & validation metrics"""
            precision_scores, recall_scores, macro_f1, weighted_f1, AUC, best_score = (
                [],
                [],
                [],
                [],
                [],
                [],
            )
            TN_train, TP_train, FP_train, FN_train = [], [], [], []

            for i, (train_idx, valid_idx) in enumerate(
                self.skf.split(train_X, train_y)
            ):
                # Train & validation indexes
                fold_train_X, fold_valid_X = (
                    train_X.iloc[train_idx],
                    train_X.iloc[valid_idx],
                )
                fold_train_y, fold_valid_y = (
                    train_y.iloc[train_idx],
                    train_y.iloc[valid_idx],
                )
                fold_pred_y_train = self.pipeline.predict(fold_train_X)

                """classification report of train set"""
                df = pd.DataFrame(
                    classification_report(
                        y_true=fold_train_y,
                        y_pred=fold_pred_y_train,
                        output_dict=1,
                        target_names=["non-toxic", "toxic"],
                    )
                ).transpose()

                # Precision
                precision_scores.append(np.round(df.loc["toxic", "precision"], 2))
                # Recall
                recall_scores.append(np.round(df.loc["toxic", "recall"], 2))
                # Macro f1
                macro_f1.append(np.round(df.loc["macro avg", "f1-score"], 2))
                # Weighted f1
                weighted_f1.append(np.round(df.loc["weighted avg", "f1-score"], 2))
                # Best Score
                best_score.append(self.pipeline.best_score_)
                # AUC
                AUC.append(roc_auc_score(fold_train_y, fold_pred_y_train))
                # Confusion matrix
                conf_matrix = confusion_matrix(fold_train_y, fold_pred_y_train)
                TN_train.append(conf_matrix[0][0])
                TP_train.append(conf_matrix[1][1])
                FP_train.append(conf_matrix[0][1])
                FN_train.append(conf_matrix[1][0])

            # Compute the mean of the metrics
            mean_precision = sum(precision_scores) / len(precision_scores)
            mean_recall = sum(recall_scores) / len(recall_scores)
            mean_macro_f1 = sum(macro_f1) / len(macro_f1)
            mean_weighted_f1 = sum(weighted_f1) / len(weighted_f1)
            mean_AUC_train = round((sum(AUC) / len(AUC)), 2)
            mean_best_score_train = sum(best_score) / len(best_score)
            mean_TN_train = int(sum(TN_train) / len(TN_train))
            mean_TP_train = int(sum(TP_train) / len(TP_train))
            mean_FP_train = int(sum(FP_train) / len(FP_train))
            mean_FN_train = int(sum(FN_train) / len(FN_train))

            """Show train metrics"""
            # TODO: how to insert classification_report of cross validation? I have no train_y, pred_y, and can't use mean
            # logger.info(
            #     f"\n{pd.DataFrame(classification_report(train_y, pred_y, output_dict=1, target_names=['non-toxic', 'toxic'])).transpose()}"
            # )
            logger.info(f"\n    Area Under the Curve score: {mean_AUC_train}")
            logger.info(
                f"\n Correlation matrix"
                f"\n (true negatives, false positives)\n (false negatives, true positives)"
                f"\n {mean_TN_train, mean_FP_train}\n {mean_FN_train, mean_TP_train}"
            )
            logger.info("Training Complete. Logging train results into MLFlow")

            """Log train metrics"""
            # Precision
            mlflow.log_metric("Precision", mean_precision)

            # Recall
            mlflow.log_metric("Recall", mean_recall)

            # macro_f1
            mlflow.log_metric("F1_macro", mean_macro_f1)

            # weighted_f1
            mlflow.log_metric("F1_weighted", mean_weighted_f1)

            # Best Score
            mlflow.log_metric("Best Score", "%.2f " % mean_best_score_train)

            # AUC
            mlflow.log_metric("AUC", mean_AUC_train)

            # Confusion matrix
            mlflow.log_metric("Conf_TN", mean_TN_train)
            mlflow.log_metric("Conf_TP", mean_TP_train)
            mlflow.log_metric("Conf_FP", mean_FP_train)
            mlflow.log_metric("Conf_FN", mean_FN_train)

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

            # toxicity threshold
            mlflow.log_param("tox_threshold", self.tox_threshold)

            """log model type"""
            mlflow.set_tag("Model", self.classifier_type)

        """Predict valid metrics and save to mlflow"""
        with mlflow.start_run():
            mlflow.set_tag(
                "mlflow.runName", f"valid_{mlflow.active_run().info.run_name}"
            )

            """Get mean validation metrics"""
            precision_scores, recall_scores, macro_f1, weighted_f1, AUC, best_score = (
                [],
                [],
                [],
                [],
                [],
                [],
            )
            TN_valid, TP_valid, FP_valid, FN_valid = [], [], [], []

            for i, (train_idx, valid_idx) in enumerate(
                self.skf.split(train_X, train_y)
            ):
                # Train & validation indexes
                fold_train_X, fold_valid_X = (
                    train_X.iloc[train_idx],
                    train_X.iloc[valid_idx],
                )
                fold_train_y, fold_valid_y = (
                    train_y.iloc[train_idx],
                    train_y.iloc[valid_idx],
                )
                fold_pred_y_valid = self.pipeline.predict(fold_valid_X)

                """classification report of valid set"""
                df = pd.DataFrame(
                    classification_report(
                        y_true=fold_valid_y,
                        y_pred=fold_pred_y_valid,
                        output_dict=1,
                        target_names=["non-toxic", "toxic"],
                    )
                ).transpose()

                # Precision
                precision_scores.append(np.round(df.loc["toxic", "precision"], 2))
                # Recall
                recall_scores.append(np.round(df.loc["toxic", "recall"], 2))
                # Macro f1
                macro_f1.append(np.round(df.loc["macro avg", "f1-score"], 2))
                # Weighted f1
                weighted_f1.append(np.round(df.loc["weighted avg", "f1-score"], 2))
                # Best Score
                best_score.append(self.pipeline.best_score_)
                # AUC
                AUC.append(roc_auc_score(fold_valid_y, fold_pred_y_valid))
                # Confusion matrix
                conf_matrix = confusion_matrix(fold_valid_y, fold_pred_y_valid)
                TN_valid.append(conf_matrix[0][0])
                TP_valid.append(conf_matrix[1][1])
                FP_valid.append(conf_matrix[0][1])
                FN_valid.append(conf_matrix[1][0])

            # Compute the mean of the metrics
            mean_precision = sum(precision_scores) / len(precision_scores)
            mean_recall = sum(recall_scores) / len(recall_scores)
            mean_macro_f1 = sum(macro_f1) / len(macro_f1)
            mean_weighted_f1 = sum(weighted_f1) / len(weighted_f1)
            mean_AUC = round((sum(AUC) / len(AUC)), 2)
            mean_best_score = sum(best_score) / len(best_score)
            mean_TN_valid = int(sum(TN_valid) / len(TN_valid))
            mean_TP_valid = int(sum(TP_valid) / len(TP_valid))
            mean_FP_valid = int(sum(FP_valid) / len(FP_valid))
            mean_FN_valid = int(sum(FN_valid) / len(FN_valid))

            logger.info("↓↓↓ VALID METRICS ↓↓↓")
            """Show valid metrics"""
            # TODO: how to insert classification_report of cross validation? I have no train_y, pred_y, and can't use mean
            # logger.info(
            #     f"\n{pd.DataFrame(classification_report(train_y, pred_y, output_dict=1, target_names=['non-toxic', 'toxic'])).transpose()}"
            # )
            logger.info(f"\n    Area Under the Curve score: {mean_AUC}")
            logger.info(
                f"\n Correlation matrix"
                f"\n (true negatives, false positives)\n (false negatives, true positives)"
                f"\n {mean_TN_valid, mean_FP_valid}\n {mean_FN_valid, mean_TP_valid}"
            )
            logger.info("Logging validation results into MLFlow")

            """Log valid metrics"""
            # Precision
            mlflow.log_metric("Precision", mean_precision)

            # Recall
            mlflow.log_metric("Recall", mean_recall)

            # macro_f1
            mlflow.log_metric("F1_macro", mean_macro_f1)

            # weighted_f1
            mlflow.log_metric("F1_weighted", mean_weighted_f1)

            # Best Score
            mlflow.log_metric("Best Score", "%.2f " % mean_best_score)

            # AUC
            mlflow.log_metric("AUC", mean_AUC)

            # Confusion matrix
            mlflow.log_metric("Conf_TN", mean_TN_valid)
            mlflow.log_metric("Conf_TP", mean_TP_valid)
            mlflow.log_metric("Conf_FP", mean_FP_valid)
            mlflow.log_metric("Conf_FN", mean_FN_valid)

        """Predict test metrics and save to mlflow"""
        with mlflow.start_run():
            mlflow.set_tag(
                "mlflow.runName", f"test_{mlflow.active_run().info.run_name}"
            )
            test_X, test_y = self.test_X, self.test_y
            """Predict on test data"""
            pred_y = self.pipeline.predict(test_X)
            # self.pipeline.save()

            """classification report of train set"""
            df = pd.DataFrame(
                classification_report(
                    y_true=test_y,
                    y_pred=pred_y,
                    output_dict=1,
                    target_names=["non-toxic", "toxic"],
                )
            ).transpose()

            """Show test metrics"""
            logger.info("↓↓↓ TEST METRICS ↓↓↓")
            logger.info(
                f"\n{pd.DataFrame(classification_report(test_y, pred_y, output_dict=1, target_names=['non-toxic', 'toxic'])).transpose()}"
            )
            logger.info(
                f"\n    Area Under the Curve score: {round(roc_auc_score(test_y, pred_y), 2)}"
            )
            logger.info(
                f"\n [true negatives  false positives]\n [false negatives  true positives] \
                        \n {confusion_matrix(test_y, pred_y)}"
            )

            # logger.info(" Logging results into file_log.log")
            logger.info("Logging test results into MLFlow")

            """Log train metrics"""
            # Precisionpycharm
            mlflow.log_metric("Precision", np.round(df.loc["toxic", "precision"], 2))

            # Recall
            mlflow.log_metric("Recall", np.round(df.loc["toxic", "recall"], 2))

            # macro_f1
            mlflow.log_metric("F1_macro", np.round(df.loc["macro avg", "f1-score"], 2))

            # weighted_f1
            mlflow.log_metric(
                "F1_weighted", np.round(df.loc["weighted avg", "f1-score"], 2)
            )

            # Best Score
            mlflow.log_metric("Best Score", "%.2f " % self.pipeline.best_score_)

            # AUC
            mlflow.log_metric("AUC", round(roc_auc_score(test_y, pred_y), 2))

            # Confusion matrix
            conf_matrix = confusion_matrix(test_y, pred_y)
            mlflow.log_metric("Conf_TN", conf_matrix[0][0])
            mlflow.log_metric("Conf_TP", conf_matrix[1][1])
            mlflow.log_metric("Conf_FP", conf_matrix[0][1])
            mlflow.log_metric("Conf_FN", conf_matrix[1][0])

        if self.save_model:
            """Log(save) model"""
            mlflow.sklearn.log_model(
                self.pipeline,
                artifact_path="/Method_1_standart/data/model_log",
                serialization_format="pickle",
            )
            # mlflow.set_tag("Version", run_version)
            logger.info("Model Trained and saved into MLFlow artifact location")
        else:
            logger.info("Model Trained but not saved into MLFlow artifact location")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--path",
        help="Data path",
        # default="../../../DB's/Toxic_database/tox_train.csv",  # Home-PC
        default="D:/Programming/db's/toxicity_main/tox_train.csv",  # Work-PC
    )
    parser.add_argument("--n_samples", help="How many samples to pass?", default=10000)
    parser.add_argument(
        "--tox_threshold",
        help="What's a threshold for toxicity from 0.0 to 1.0?",
        default=0.475,
    )
    parser.add_argument(
        "--n_trials", help="How many trials for hyperparameter tuning?", default=1
    )
    parser.add_argument(
        "--type_of_run", help='"train" of "inference"?', default="train"
    )
    parser.add_argument(
        "--vectorizer", help='Choose "tfidf" or "spacy"', default="tfidf"
    )
    parser.add_argument(
        "--classifier_type",
        help='Choose "logreg", "xgboost" or "lgbm"',
        default="logreg",
    )
    parser.add_argument(
        "--save_model",
        help="Choose True or False",
        default=False,
    )
    args = parser.parse_args()
    if args.type_of_run == "train":
        classifier = ClassifierModel(
            input_data=args.path,
            n_samples=args.n_samples,
            tox_threshold=args.tox_threshold,
            n_trials=args.n_trials,
            vectorizer=args.vectorizer,
            classifier_type=args.classifier_type,
            save_model=args.save_model,
        )
        classifier.train()

    # if args.type_of_run=='inference':
    #     with open('data/tfidf.pk', 'r') as f:
    #         tfidf=pickle.load(f)
    #     import model in pickle too
    #     classifier = ClassifierModel()  # put inference params
    #     classifier.predict()
