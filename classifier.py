import argparse
import pickle
import mlflow.sklearn
from mlflow import log_metric, log_param, log_artifacts
import pandas as pd
import numpy as np
import os
import warnings
from sklearn.pipeline import make_pipeline
from src.utils.features_preprocessing import Preprocessor
from src.utils.utils import ReadPrepare, Split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import logging as logger
from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna
from optuna.integration import OptunaSearchCV
from xgboost import XGBClassifier
import lightgbm as lgb

warnings.filterwarnings('ignore')
logger.getLogger().setLevel(logger.INFO)
# logger.basicConfig(filename='log_file.log',
#                     encoding='utf-8',filemode='w')
# path = "../../DB's/Toxic_database/tox_train.csv"  # relative path
# path = "D:/Programming/DB's/Toxic_database/tox_train.csv"  # absolute path

RANDOM_STATE = 42


class ClassifierModel:
    def __init__(self,
                 input_data,
                 n_samples=800,
                 tfidf_max_feat=500,
                 classifier_type: str = 'basemodel',
                 vectorizer: str = 'tfidf',
                 pipeline=None):
        self.input_data = input_data
        self.n_samples = n_samples
        self.max_feat = tfidf_max_feat
        self.classifier_type = classifier_type
        self.vectorizer = vectorizer
        self.pipeline = pipeline
        self.x_train = self.y_train = self.x_test = self.y_test = None

    def __training_setup(self):
        """ReadPrepare & Split parts"""
        df = ReadPrepare(self.input_data, self.n_samples).data_process()
        self.train_X, self.train_y = Split(df=df).get_train_data()
        self.test_X, self.test_y = Split(df=df).get_test_data()

        """Vectorizer"""
        pass

        """Model pipeline"""
        if self.classifier_type == 'basemodel':
            svc_pipeline = make_pipeline(Preprocessor(), SVC(random_state=RANDOM_STATE, kernel='linear', degree=1))
            """init hyper-param"""
            param_distributions = {"svc__C": optuna.distributions.FloatDistribution(1e-10, 1e10)}  # ,
            # "svc__gamma":optuna.distributions.FloatDistribution(1e-4,1)}
            self.pipeline = OptunaSearchCV(svc_pipeline, param_distributions,
                                           cv=StratifiedKFold(n_splits=3, shuffle=True),
                                           n_trials=1, random_state=42, verbose=0,scoring=None)

        elif self.classifier_type == "xgboost":
            xgb_pipeline = make_pipeline(Preprocessor(), XGBClassifier())  # colsample_bytree=0.7, learning_rate=0.05,
            #                                                         max_depth=5,min_child_weight=11, n_estimators=1000,
            #                                                         n_jobs=4,objective='binary:multiclass',
            #                                                         random_state=RANDOM_STATE, subsample=0.8))
            """init hyper-param"""
            param_distributions = {}
            # param_distributions = {
            #     'clf__njobs': [4],
            #     'clf__objective': ['multiclass'],
            #     'clf__learning_rate': [0.05],
            #     'clf__max_depth': [6, 12, 18],
            #     'clf__min_child_weight': [11, 13, 15],
            #     'clf__subsample': [0.7, 0.8],
            #     'clf__colsample_bytree': [0.6, 0.7],
            #     'clf__n_estimators': [5, 50, 100, 1000],
            #     'clf__missing': [-999],
            #     'clf__random_state': [RANDOM_STATE]
            # }
            self.pipeline = OptunaSearchCV(xgb_pipeline, param_distributions,
                                           cv=StratifiedKFold(n_splits=3, shuffle=True),
                                           n_trials=1, random_state=42, verbose=0)

        elif self.classifier_type == 'lgbm':
            lgbm_pipeline = make_pipeline(Preprocessor(), lgb.LGBMClassifier())
            """init hyper-param"""
            param_distributions = {}
            # param_distributions = {'num_leaves': 5,
            #           'objective': 'multiclass',
            #           'num_class': len(np.unique(self.y_train)),
            #           'learning_rate': 0.01,
            #           'max_depth': 5,
            #           'random_state': RANDOM_STATE}
            self.pipeline = OptunaSearchCV(lgbm_pipeline, param_distributions,
                                           cv=StratifiedKFold(n_splits=3, shuffle=True),
                                           n_trials=1, random_state=42, verbose=0)

    def train(self, run_version=None):
        """... and one method to rule them all. (c)"""
        self.__training_setup()
        train_X, train_y = self.train_X, self.train_y
        """MLFlow Config"""
        logger.info("Setting up MLFlow Config")
        mlflow.set_experiment('Toxicity_classifier_model')
        """Search for previous runs and get run_id if present"""
        # logger.info("Searching for previous runs for given model type")
        # df_runs = mlflow.search_runs(filter_string="tags.Model = '{0}'".format('XGB'))
        # df_runs = df_runs.loc[~df_runs['tags.Version'].isna(), :] if 'tags.Version' in df_runs else pd.DataFrame()
        # run_id = df_runs.loc[df_runs['tags.Version'] == run_version, 'run_id'].iloc[0]
        # run_id =3
        # load_prev = True
        # run_version = len(df_runs) + 1

        """Start the MLFlow Run and train the model"""
        with mlflow.start_run(run_id=None):
            """Train & predict"""
            self.pipeline.fit(train_X, train_y)
            logger.info("train is done")
            pred_y = self.pipeline.predict(train_X)
            # self.pipeline.save()
            """classification report on train set"""
            df = pd.DataFrame(classification_report(y_true=train_y, y_pred=pred_y, output_dict=1,
                                                    target_names=['non-toxic', 'toxic'])).transpose()
            # logger.info(f"\n\nTrain metric.")
            # print(df)

            """cross_val_score(nested_CV) on train set"""
            # kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            # score=cross_val_score(self.pipeline,train_X,train_y,cv=StratifiedKFold(n_splits=3, shuffle=True))
            # logger.info(f"cross_val_score : {score.mean()}")

            """Log model metrics"""
            logger.info(
                f"\n {pd.DataFrame(classification_report(train_y, pred_y, output_dict=1, target_names=['non-toxic', 'toxic'])).transpose()}")
            logger.info(f"Area Under the Curve score: {round(roc_auc_score(train_y, pred_y), 2)}")
            logger.info(f"\n [true negatives  false positives]\n [false negatives  true positives] \
                        \n {confusion_matrix(train_y, pred_y)}")

            # logger.info(" Logging results into file_log.log")
            logger.info("Training Complete. Logging results into MLFlow")
            mlflow.log_metric("macro_f1", np.round(df.loc["macro avg", "f1-score"], 5))
            mlflow.log_metric("weighted_f1", np.round(df.loc["weighted avg", "f1-score"], 5))
            df = df.reset_index()
            df.columns = ['category', 'precision', 'recall', 'f1-score', 'support']
            # df.to_csv("toxicity_full_report.csv")
            # mlflow.log_artifact("toxicity_full_report.csv")
            mlflow.log_param("Best Params", {k: round(v, 2) for k, v in self.pipeline.best_params_.items()})
            mlflow.log_param("Best Score", "%.2f " % self.pipeline.best_score_)
            # os.remove("toxicity_full_report.csv")
            """Save model"""
            mlflow.sklearn.log_model(self.pipeline,
                                     artifact_path='D:\Programming\Repositories\\toxic_detection_classification\data\model_log',
                                     serialization_format='pickle')
            mlflow.set_tag("Model", self.classifier_type)
            # mlflow.set_tag("Version", run_version)
            logger.info("Model Trained and saved into MLFlow artifact location")

            """Test set metrics """
            # predict_y = pipeline.predict(test_X)
            # logger.info(f"\n\nTest metrics.")
            # logger.info(classification_report(test_y, pred_y, target_names=['Not_toxic', 'Toxic']))
            # logger.info(f"Area Under the Curve score: {round(roc_auc_score(test_y, pred_y), 2)}")
            # logger.info(f"\n [true negatives  false negatives]\n [true positives  false positives]")
            # logger.info(f"\n{confusion_matrix(test_y, pred_y)}")
            # logger.info("Testing completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path',
                        help='Data path',
                        default="../../DB's/Toxic_database/tox_train.csv")
    parser.add_argument('--n_samples',
                        help='How many samples to pass?',
                        default=800)
    parser.add_argument('--type_of_run',
                        help='"train" of "inference"?',
                        default='train')
    parser.add_argument('--classifier_type',
                        help='Choose "basemodel", "xgboost" or "lightgbm"',
                        default='lgbm')
    parser.add_argument('--vectorizer',
                        help='Choose "tfidf" or "spacy"',
                        default='tfidf')
    args = parser.parse_args()

    if args.type_of_run == 'train':
        classifier = ClassifierModel(input_data=args.path, n_samples=args.n_samples,
                                     classifier_type=args.classifier_type,vectorizer=args.vectorizer)
        classifier.train()

    # if args.type_of_run=='inference':
    #     with open('data/tfidf.pk', 'r') as f:
    #         tfidf=pickle.load(f)
    #     import model in pickle too
    #     classifier = ClassifierModel()  # put inference params
    #     classifier.predict()
