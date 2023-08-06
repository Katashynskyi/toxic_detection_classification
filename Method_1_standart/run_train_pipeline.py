import argparse
import logging as logger
import warnings
import os
import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
from optuna.integration import OptunaSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.utils.features_preprocessing import Preprocessor
from src.utils.utils import ReadPrepare, Split


# Preset settings
RANDOM_STATE = 42
pd.set_option("display.width", 1000)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
warnings.filterwarnings("ignore")
logger.getLogger().setLevel(logger.INFO)


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
        self.train_x, self.train_y = Split(df=df).get_train_data()
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
        train_x, train_y = self.train_x, self.train_y

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
            self.pipeline.fit(train_x, train_y)
            logger.info("train is done")
            logger.info("↓↓↓ TRAIN METRICS ↓↓↓")

            """Get mean train & validation metrics"""
            precision_scores, recall_scores, macro_f1, weighted_f1, auc, best_score = (
                [],
                [],
                [],
                [],
                [],
                [],
            )
            tn_train, tp_train, fp_train, fn_train = [], [], [], []

            for i, (train_idx, valid_idx) in enumerate(
                self.skf.split(train_x, train_y)
            ):
                # Train & validation indexes
                fold_train_x, fold_valid_x = (
                    train_x.iloc[train_idx],
                    train_x.iloc[valid_idx],
                )
                fold_train_y, fold_valid_y = (
                    train_y.iloc[train_idx],
                    train_y.iloc[valid_idx],
                )
                fold_pred_y_train = self.pipeline.predict(fold_train_x)

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
                auc.append(roc_auc_score(fold_train_y, fold_pred_y_train))
                # Confusion matrix
                conf_matrix = confusion_matrix(fold_train_y, fold_pred_y_train)
                tn_train.append(conf_matrix[0][0])
                tp_train.append(conf_matrix[1][1])
                fp_train.append(conf_matrix[0][1])
                fn_train.append(conf_matrix[1][0])

            # Compute the mean of the metrics
            mean_precision = sum(precision_scores) / len(precision_scores)
            mean_recall = sum(recall_scores) / len(recall_scores)
            mean_macro_f1 = sum(macro_f1) / len(macro_f1)
            mean_weighted_f1 = sum(weighted_f1) / len(weighted_f1)
            mean_auc_train = round((sum(auc) / len(auc)), 2)
            mean_best_score_train = sum(best_score) / len(best_score)
            mean_tn_train = int(sum(tn_train) / len(tn_train))
            mean_tp_train = int(sum(tp_train) / len(tp_train))
            mean_fp_train = int(sum(fp_train) / len(fp_train))
            mean_fn_train = int(sum(fn_train) / len(fn_train))

            """Show train metrics"""
            # TODO: how to insert classification_report of cross validation?
            #  I have no train_y, pred_y, and can't use mean
            # logger.info(
            #     f"\n{pd.DataFrame(classification_report(train_y, pred_y, output_dict=1, target_names=['non-toxic', 'toxic'])).transpose()}"
            # )
            logger.info(f"\n    Area Under the Curve score: {mean_auc_train}")
            logger.info(
                f"\n Correlation matrix"
                f"\n (true negatives, false positives)\n (false negatives, true positives)"
                f"\n {mean_tn_train, mean_fp_train}\n {mean_fn_train, mean_tp_train}"
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
            mlflow.log_metric("AUC", mean_auc_train)

            # Confusion matrix
            mlflow.log_metric("Conf_TN", mean_tn_train)
            mlflow.log_metric("Conf_TP", mean_tp_train)
            mlflow.log_metric("Conf_FP", mean_fp_train)
            mlflow.log_metric("Conf_FN", mean_fn_train)

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

            # number of trials of hyperparameter tuning
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
            precision_scores, recall_scores, macro_f1, weighted_f1, auc, best_score = (
                [],
                [],
                [],
                [],
                [],
                [],
            )
            tn_valid, tp_valid, fp_valid, fn_valid = [], [], [], []

            for i, (train_idx, valid_idx) in enumerate(
                self.skf.split(train_x, train_y)
            ):
                # Train & validation indexes
                fold_train_x, fold_valid_x = (
                    train_x.iloc[train_idx],
                    train_x.iloc[valid_idx],
                )
                fold_train_y, fold_valid_y = (
                    train_y.iloc[train_idx],
                    train_y.iloc[valid_idx],
                )
                fold_pred_y_valid = self.pipeline.predict(fold_valid_x)

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
                auc.append(roc_auc_score(fold_valid_y, fold_pred_y_valid))
                # Confusion matrix
                conf_matrix = confusion_matrix(fold_valid_y, fold_pred_y_valid)
                tn_valid.append(conf_matrix[0][0])
                tp_valid.append(conf_matrix[1][1])
                fp_valid.append(conf_matrix[0][1])
                fn_valid.append(conf_matrix[1][0])

            # Compute the mean of the metrics
            mean_precision = sum(precision_scores) / len(precision_scores)
            mean_recall = sum(recall_scores) / len(recall_scores)
            mean_macro_f1 = sum(macro_f1) / len(macro_f1)
            mean_weighted_f1 = sum(weighted_f1) / len(weighted_f1)
            mean_auc = round((sum(auc) / len(auc)), 2)
            mean_best_score = sum(best_score) / len(best_score)
            mean_tn_valid = int(sum(tn_valid) / len(tn_valid))
            mean_tp_valid = int(sum(tp_valid) / len(tp_valid))
            mean_fp_valid = int(sum(fp_valid) / len(fp_valid))
            mean_fn_valid = int(sum(fn_valid) / len(fn_valid))

            logger.info("↓↓↓ VALID METRICS ↓↓↓")
            """Show valid metrics"""
            # TODO: how to insert classification_report of cross validation?
            #  I have no train_y, pred_y, and can't use mean
            # logger.info(
            #     f"\n{pd.DataFrame(classification_report(train_y, pred_y, output_dict=1, target_names=['non-toxic', 'toxic'])).transpose()}"
            # )
            logger.info(f"\n    Area Under the Curve score: {mean_auc}")
            logger.info(
                f"\n Correlation matrix"
                f"\n (true negatives, false positives)\n (false negatives, true positives)"
                f"\n {mean_tn_valid, mean_fp_valid}\n {mean_fn_valid, mean_tp_valid}"
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
            mlflow.log_metric("AUC", mean_auc)

            # Confusion matrix
            mlflow.log_metric("Conf_TN", mean_tn_valid)
            mlflow.log_metric("Conf_TP", mean_tp_valid)
            mlflow.log_metric("Conf_FP", mean_fp_valid)
            mlflow.log_metric("Conf_FN", mean_fn_valid)

        """Predict test metrics and save to mlflow"""
        with mlflow.start_run():
            mlflow.set_tag(
                "mlflow.runName", f"test_{mlflow.active_run().info.run_name}"
            )
            test_x, test_y = self.test_X, self.test_y
            """Predict on test data"""
            pred_y = self.pipeline.predict(test_x)
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
            try:
                os.mkdir("./data/")
            except:
                pass
            """Log(save) model"""
            import pickle

            # Define the path to save the model
            model_path = "data/classic_model.pkl"
            # Save the model to a pickle file
            with open(model_path, "wb") as model_file:
                pickle.dump(self.pipeline, model_file)
            logger.info("Model Trained and saved into MLFlow artifact location")
        else:
            logger.info("Model Trained but not saved into MLFlow artifact location")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--path",
        help="Data path",
        default="../../../DB's/Toxic_database/tox_train.csv",  # Home-PC
        # default="D:/Programming/db's/toxicity_main/tox_train.csv",  # Work-PC
    )
    parser.add_argument("--n_samples", help="How many samples to pass?", default=500)
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
        "--vectorizer", help='Choose "tfidf" or "spacy"', default="spacy"
    )
    parser.add_argument(
        "--classifier_type",
        help='Choose "logreg", "xgboost" or "lgbm"',
        default="lgbm",
    )
    parser.add_argument(
        "--save_model",
        help="Choose True or False",
        default=True,
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
