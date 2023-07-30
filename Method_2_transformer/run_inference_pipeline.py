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
from src.utils.utils import ReadPrepare
from src.utils.utils_bert import DistilBERTClass
from src.utils.utils_dataset import MultiLabelDataset
from src.utils.utils_metrics import log_metrics
from src.utils.train_test_valid_runs import RunTrainValidTest

# from sklearn import metrics
import warnings

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
        inference_batch_size,
        learning_rate,
        threshold,
        num_classes,
    ):
        self.path = path
        self.n_samples = n_samples
        self.random_state = random_state
        self.max_len = max_len
        self.inference_batch_size = inference_batch_size
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.num_classes = num_classes

        self.inference_targets = self.inference_outputs = 0

    def __training_setup(self):
        """hidden method to prepare dataset.
        split it, prepare weights formula, initialize tokenizer, transform to custom pytorch datasets
        """
        # ReadPrepare df csv file
        df = ReadPrepare(
            path=self.path, n_samples=self.n_samples
        ).data_process()  # csv -> pd.DataFrame
        df.reset_index()

        # Tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased", truncation=True, do_lower_case=True
        )

        # Custom pytorch datasets
        train_set = MultiLabelDataset(df, tokenizer, self.max_len)

        inference_params = {
            "batch_size": self.inference_batch_size,
            "shuffle": False,
            # 'num_workers': 8
        }

        # Iterators of datasets
        self.inference_loader = DataLoader(train_set, **inference_params)

        # Load model
        self.model = DistilBERTClass(num_classes=self.num_classes).load().to(DEVICE)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=self.learning_rate
        )

    def inference(self):
        """method to evaluate metrics
        gets pre-trained model, it's inference+metrics
        """
        self.__training_setup()

        """Evaluate test subset"""
        outputs, self.inference_targets = RunTrainValidTest(
            model=self.model, loader=self.inference_loader, optimizer=self.optimizer
        ).run_test()
        self.inference_outputs = np.array(outputs) >= self.threshold  # threshold

        mlflow.set_experiment("Toxicity_transformer_classifier")

        """Evaluate and log test metrics to mlflow"""
        with mlflow.start_run():
            mlflow.set_tag(
                "mlflow.runName", f"test_{mlflow.active_run().info.run_name}"
            )
            """"Evaluate and log test metrics"""
            log_metrics(self.inference_targets, self.inference_outputs, label="test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(  # TODO: should we choose between them or make 2 separate files?
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
        # default=128  # home_PC
        default=512,  # work_PC
    )
    # parser.add_argument("--train_batch_size", help="Train batch size", default=16)
    parser.add_argument(
        "--inference_batch_size", help="Inference batch size", default=16
    )
    # parser.add_argument("--epochs", help="Number of epochs", default=0)
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
            # train_batch_size=args.train_batch_size,
            inference_batch_size=args.inference_batch_size,
            # epochs=args.epochs,
            learning_rate=args.learning_rate,
            threshold=args.threshold,
            num_classes=args.num_classes,
        )
        classifier.inference()
