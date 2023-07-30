import argparse

# import logging as logger
import numpy as np
import pandas as pd
import torch
import mlflow

# from mlflow import log_artifacts, log_metric, log_param
# from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from src.utils.utils import Split, ReadPrepare
from src.utils.utils_bert import DistilBERTClass
from src.utils.utils_dataset import MultiLabelDataset
from src.utils.utils_metrics import log_metrics
from src.utils.train_test_valid_runs import RunTrainValidTest

# from sklearn import metrics
import warnings
from collections import Counter

# import seaborn as sns
# import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

desired_width = 1000
pd.set_option("display.width", desired_width)
pd.set_option("display.max_columns", 100)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(DEVICE)

RANDOM_STATE = 42


class TransformerModel:
    def __init__(
        self,
        path,
        n_samples,
        random_state,
        max_len,
        train_batch_size,
        valid_batch_size,
        epochs,
        learning_rate,
        threshold,
        num_classes,
    ):
        self.path = path
        self.n_samples = n_samples
        self.random_state = random_state
        self.max_len = max_len
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.num_classes = num_classes

        self.train_outputs = self.valid_outputs = self.test_outputs = 0
        self.train_targets = self.valid_targets = self.test_targets = 0

    def __training_setup(self):
        """hidden method to prepare dataset.
        split it, prepare weights formula, initialize tokenizer, transform to custom pytorch datasets
        """
        # ReadPrepare train & test csv files
        rp = ReadPrepare(
            path=self.path, n_samples=self.n_samples
        ).data_process()  # csv -> pd.DataFrame

        # Split df on train & valid & test
        splitter = Split(df=rp, test_size=0.3)
        train_data = splitter.get_train_data().reset_index()  # -> pd.DataFrame
        valid_data = splitter.get_valid_data().reset_index()  # -> pd.DataFrame
        test_data = splitter.get_test_data().reset_index()  # -> pd.DataFrame

        total_samples = len(train_data["labels"].values)
        num_classes = len(train_data["labels"][0])

        # Setup formula four weights
        weights = []
        for i in range(6):
            counts = Counter(train_data["labels"].apply(lambda x: x[i]))
            class_samples, rest = sorted(list(map(int, counts.values())))
            weight = total_samples / (num_classes * class_samples)
            weights.append(round(weight, 4))
        self.weights = torch.tensor([weights]).to(DEVICE)

        # Tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased", truncation=True, do_lower_case=True
        )

        # Custom pytorch datasets
        train_set = MultiLabelDataset(train_data, tokenizer, self.max_len)
        valid_set = MultiLabelDataset(valid_data, tokenizer, self.max_len)
        test_set = MultiLabelDataset(
            test_data, tokenizer, self.max_len
        )  # , new_data=True)

        train_params = {
            "batch_size": self.train_batch_size,
            "shuffle": True,
            # 'num_workers': 8
        }

        val_params = {
            "batch_size": self.valid_batch_size,
            "shuffle": False,
            # 'num_workers': 8
        }
        # Iterators of datasets
        self.training_loader = DataLoader(train_set, **train_params)
        self.valid_loader = DataLoader(valid_set, **val_params)
        self.test_loader = DataLoader(test_set, **val_params)
        # Init model
        self.model = DistilBERTClass(num_classes=self.num_classes).to(DEVICE)
        # Optimizer
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=self.learning_rate
        )

    def train(self):
        """train data and save model"""
        self.__training_setup()

        """fit and evaluate train subset"""
        for epoch in range(self.epochs):
            print("epoch", epoch)
            RunTrainValidTest(
                model=self.model,
                loader=self.training_loader,
                weights=self.weights,
                optimizer=self.optimizer,
            ).run_train(epoch)
        outputs, self.train_targets = RunTrainValidTest(
            model=self.model,
            loader=self.training_loader,
            weights=self.weights,
            optimizer=self.optimizer,
        ).run_train(epoch=self.epochs)

        """Toxicity threshold"""
        self.train_outputs = np.array(outputs) >= self.threshold

        """Save model"""
        # self.model._save(
        #     epoch=self.epochs, model=self.model, optimizer=self.optimizer
        # )  # TODO: should we save with ability to train again???
        self.model.save()

    def predict(self):
        """Method to predict valid data and test data"""

        """Evaluate validation subset"""
        outputs, self.valid_targets = RunTrainValidTest(
            model=self.model, loader=self.valid_loader, optimizer=self.optimizer
        ).run_validation()
        self.valid_outputs = np.array(outputs) >= self.threshold  # threshold

        """Evaluate test subset"""
        outputs, self.test_targets = RunTrainValidTest(
            model=self.model, loader=self.test_loader, optimizer=self.optimizer
        ).run_test()
        self.test_outputs = np.array(outputs) >= self.threshold  # threshold

    def evaluate(self, type_="train"):
        """method to evaluate and log train, valid, test subsample metrics
        can get pre-trained model, and it's almost inference+metrics

        Args:
            type_: "train", "valid" or "test"
        """
        mlflow.set_experiment("Toxicity_transformer_classifier")
        if type_ == "train":
            """Evaluate and log train metrics, log hyperparams to mlflow"""
            with mlflow.start_run():
                mlflow.set_tag(
                    "mlflow.runName", f"train_{mlflow.active_run().info.run_name}"
                )
                """"Evaluate and log train metrics"""
                log_metrics(self.train_targets, self.train_outputs, label="train")

                """Log hyperparams"""
                mlflow.log_param("n_samples", self.n_samples)
                mlflow.log_param("epochs", self.epochs)
                mlflow.log_param("threshold", self.threshold)
                mlflow.log_param("max_len", self.max_len)
                mlflow.log_param("learning_rate", self.learning_rate)
                mlflow.log_param("weights", self.weights)

        elif type_ == "valid":
            """Evaluate and log valid metrics to mlflow"""
            with mlflow.start_run():
                mlflow.set_tag(
                    "mlflow.runName", f"valid_{mlflow.active_run().info.run_name}"
                )
                """"Evaluate and log valid metrics"""
                log_metrics(self.valid_targets, self.valid_outputs, label="valid")

        elif type_ == "test":
            """Evaluate and log test metrics to mlflow"""
            with mlflow.start_run():
                mlflow.set_tag(
                    "mlflow.runName", f"test_{mlflow.active_run().info.run_name}"
                )
                """"Evaluate and log test metrics"""
                log_metrics(self.test_targets, self.test_outputs, label="test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--type_of_run", help='"train" of "inference"?', default="train"
    )
    parser.add_argument(
        "--path",
        help="Data path",
        default="D:/Programming/DB's/toxic_db_for_transformer/train.csv",  # Home-PC
        # default="D:/Programming/db's/toxicity_kaggle_1/train.csv",  # Work-PC
    )
    parser.add_argument(
        "--random_state", help="Choose seed for random state", default=RANDOM_STATE
    )
    parser.add_argument(
        # TODO: Max length of ???
        "--max_len",
        help="Max length of ???",
        default=128  # home_PC
        # default=512 # work_PC
    )
    parser.add_argument("--train_batch_size", help="Train batch size", default=16)
    parser.add_argument("--valid_batch_size", help="Valid batch size", default=16)
    parser.add_argument("--epochs", help="Number of epochs", default=0)
    parser.add_argument(
        "--learning_rate", help="Learning rate", default=1e-05
    )  # 0.001, 0.005, 0.01, 0.05, 0.1
    parser.add_argument("--n_samples", help="How many samples to pass?", default=600)
    parser.add_argument(
        "--threshold", help="What's the threshold for toxicity?", default=0.5
    )
    parser.add_argument(
        "--num_classes", help="Choose number of classes to predict", default=6
    )
    args = parser.parse_args()
    if args.type_of_run == "train":
        classifier = TransformerModel(
            path=args.path,
            n_samples=args.n_samples,
            random_state=args.random_state,
            max_len=args.max_len,
            train_batch_size=args.train_batch_size,
            valid_batch_size=args.valid_batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            threshold=args.threshold,
            num_classes=args.num_classes,
        )

        classifier.train()
        classifier.predict()
        for eval_type in ["train", "valid", "test"]:
            classifier.evaluate(type_=eval_type)
