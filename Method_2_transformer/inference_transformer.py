import argparse
import logging as logger
import numpy as np
import pandas as pd
import torch
import mlflow
from mlflow import log_artifacts, log_metric, log_param
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from src.utils.utils import Split, ReadPrepare
from src.utils.utils_bert import MultiLabelDataset, DistilBERTClass

from sklearn import metrics
import warnings
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

desired_width = 1000
pd.set_option("display.width", desired_width)
pd.set_option("display.max_columns", 100)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(DEVICE)


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

    def __training_setup(self):
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
        mlflow.set_experiment("Toxicity_transformer_classifier")

        """Train model and save train metrics to mlflow"""
        with mlflow.start_run():
            mlflow.set_tag(
                "mlflow.runName", f"train_{mlflow.active_run().info.run_name}"
            )
            self.__training_setup()
            """Train & predict"""

            def run_train(epoch):
                self.model.train()
                fin_outputs = []
                fin_targets = []
                for _, data in tqdm(enumerate(self.training_loader, 0)):
                    ids = data["ids"].to(DEVICE, dtype=torch.long)
                    mask = data["mask"].to(DEVICE, dtype=torch.long)
                    token_type_ids = data["token_type_ids"].to(DEVICE, dtype=torch.long)
                    targets = data["targets"].to(DEVICE, dtype=torch.float)
                    outputs = self.model(ids, mask, token_type_ids)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        outputs, targets, weight=self.weights
                    )
                    if _ % 5 == 0:
                        print(f"Epoch: {epoch}, Loss:  {loss.item()}")
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    fin_targets.extend(targets.cpu().detach().numpy().tolist())
                    fin_outputs.extend(
                        torch.sigmoid(outputs).cpu().detach().numpy().tolist()
                    )
                return fin_outputs, fin_targets

            for epoch in range(self.epochs):
                print("epoch", epoch)
                run_train(epoch)
            logger.info("train is done")
            outputs, targets = run_train(self.epochs)
            outputs = np.array(outputs) >= self.threshold  # threshold

            """Compute train metrics"""
            train_precision = metrics.precision_score(
                targets, outputs, average="weighted"
            )
            train_recall = metrics.recall_score(targets, outputs, average="weighted")
            train_conf_matrix = metrics.multilabel_confusion_matrix(targets, outputs)
            train_f1_score_micro = metrics.f1_score(targets, outputs, average="micro")
            train_f1_score_macro = metrics.f1_score(targets, outputs, average="macro")
            label_columns = [
                "train_toxic",
                "train_severe_toxic",
                "train_obscene",
                "train_threat",
                "train_insult",
                "train_identity_hate",
            ]
            for i, matrix in enumerate(train_conf_matrix):
                print(label_columns[i])
                print(matrix)

            conf_matrix = metrics.multilabel_confusion_matrix(targets, outputs)
            label_names = label_columns

            # visualization
            for i, matrix in enumerate(conf_matrix):
                plt.figure()
                sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
                plt.title(f"Confusion Matrix - {label_columns[i]}")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.savefig(f"conf_matrix_{i}.png")  # Save as PNG file
                plt.close()

                with open(f"conf_matrix_{i}.png", "rb") as file:
                    mlflow.log_artifact("file", artifact_path=f"confusion_matrix_{i}")
            mlflow.log_artifact("file.png")

            """Log train metrics"""
            # Precision
            mlflow.log_metric("Precision", train_precision)
            # Recall
            mlflow.log_metric("Recall", train_recall)
            # micro_f1
            mlflow.log_metric("F1_micro", train_f1_score_micro)
            # macro_f1
            mlflow.log_metric("F1_macro", train_f1_score_macro)

            """Log hyperparams"""
            mlflow.log_param("n_samples", self.n_samples)
            mlflow.log_param("epochs", self.epochs)
            mlflow.log_param("threshold", self.threshold)
            mlflow.log_param("max_len", self.max_len)
            mlflow.log_param("learning_rate", self.learning_rate)
            mlflow.log_param("weights", self.weights)

        """Predict valid metrics and save to mlflow"""
        with mlflow.start_run():
            mlflow.set_tag(
                "mlflow.runName", f"valid_{mlflow.active_run().info.run_name}"
            )

            def run_validation():
                self.model.eval()
                fin_outputs = []
                fin_targets = []
                with torch.no_grad():
                    for _, data in enumerate(self.valid_loader, 0):
                        ids = data["ids"].to(DEVICE, dtype=torch.long)
                        mask = data["mask"].to(DEVICE, dtype=torch.long)
                        token_type_ids = data["token_type_ids"].to(
                            DEVICE, dtype=torch.long
                        )
                        targets = data["targets"].to(DEVICE, dtype=torch.float)
                        outputs = self.model(ids, mask, token_type_ids)
                        fin_targets.extend(targets.cpu().detach().numpy().tolist())
                        fin_outputs.extend(
                            torch.sigmoid(outputs).cpu().detach().numpy().tolist()
                        )
                    return fin_outputs, fin_targets

            outputs, targets = run_validation()
            outputs = np.array(outputs) >= self.threshold  # threshold
            logger.info("validation is done")

            """Compute metrics"""
            precision = metrics.precision_score(targets, outputs, average="weighted")
            recall = metrics.recall_score(targets, outputs, average="weighted")
            conf_matrix = metrics.multilabel_confusion_matrix(targets, outputs)
            f1_score_micro = metrics.f1_score(targets, outputs, average="micro")
            f1_score_macro = metrics.f1_score(targets, outputs, average="macro")
            label_columns = [
                "valid_toxic",
                "valid_severe_toxic",
                "valid_obscene",
                "valid_threat",
                "valid_insult",
                "valid_identity_hate",
            ]
            for i, matrix in enumerate(conf_matrix):
                print(label_columns[i])
                print(matrix)

            """Log valid metrics"""
            # Precision
            mlflow.log_metric("Precision", float(precision))
            # Recall
            mlflow.log_metric("Recall", recall)
            # micro_f1
            mlflow.log_metric("F1_micro", f1_score_micro)
            # macro_f1
            mlflow.log_metric("F1_macro", f1_score_macro)

        """Predict test metrics and save to mlflow"""
        with mlflow.start_run():
            mlflow.set_tag(
                "mlflow.runName", f"test_{mlflow.active_run().info.run_name}"
            )

            def run_test():
                self.model.eval()
                fin_outputs = []
                fin_targets = []
                with torch.inference_mode():
                    for _, data in enumerate(self.test_loader, 0):
                        ids = data["ids"].to(DEVICE, dtype=torch.long)
                        mask = data["mask"].to(DEVICE, dtype=torch.long)
                        token_type_ids = data["token_type_ids"].to(
                            DEVICE, dtype=torch.long
                        )
                        targets = data["targets"].to(DEVICE, dtype=torch.float)
                        outputs = self.model(ids, mask, token_type_ids)
                        fin_targets.extend(targets.cpu().detach().numpy().tolist())
                        fin_outputs.extend(
                            torch.sigmoid(outputs).cpu().detach().numpy().tolist()
                        )
                    return fin_outputs, fin_targets

            outputs, targets = run_test()
            outputs = np.array(outputs) >= self.threshold  # threshold
            logger.info("test is done")

            """Compute metrics"""
            precision = metrics.precision_score(targets, outputs, average="weighted")
            recall = metrics.recall_score(targets, outputs, average="weighted")
            conf_matrix = metrics.multilabel_confusion_matrix(targets, outputs)
            f1_score_micro = metrics.f1_score(targets, outputs, average="micro")
            f1_score_macro = metrics.f1_score(targets, outputs, average="macro")
            label_columns = [
                "test_toxic",
                "test_severe_toxic",
                "test_obscene",
                "test_threat",
                "test_insult",
                "test_identity_hate",
            ]
            for i, matrix in enumerate(conf_matrix):
                print(label_columns[i])
                print(matrix)

            """Log test metrics"""
            # Precision
            mlflow.log_metric("Precision", float(precision))
            # Recall
            mlflow.log_metric("Recall", recall)
            # micro_f1
            mlflow.log_metric("F1_micro", f1_score_micro)
            # macro_f1
            mlflow.log_metric("F1_macro", f1_score_macro)

    def inference(self):
        def __training_setup(self):
            # ReadPrepare train & test csv files
            rp = ReadPrepare(
                path=self.path, n_samples=self.n_samples
            ).data_process()  # csv -> pd.DataFrame

            # Split df on train & valid & test
            # splitter = Split(df=rp, test_size=0.3)
            # get data from df
            test_data = splitter.get_test_data().reset_index()  # -> pd.DataFrame

            # total_samples = len(train_data["labels"].values)

            # Tokenizer
            # read from output model from train
            tokenizer = DistilBertTokenizer.from_pretrained(
                "distilbert-base-uncased", truncation=True, do_lower_case=True
            )

            # Custom pytorch datasets
            test_set = MultiLabelDataset(
                test_data, tokenizer, self.max_len
            )  # , new_data=True)

            test_params = {
                "batch_size": self.test_batch_size,
                "shuffle": False,
                # 'num_workers': 8
            }
            # Iterators of datasets
            self.test_loader = DataLoader(test_set, **test_params)
            # Init model
            self.model = DistilBERTClass(num_classes=self.num_classes).to(DEVICE)
            # Optimizer

        def train(self):
            mlflow.set_experiment("Toxicity_transformer_classifier")

            """Predict test metrics and save to mlflow"""

            def run_test():
                self.model.eval()
                fin_outputs = []
                with torch.inference_mode():
                    for _, data in enumerate(self.test_loader, 0):
                        ids = data["ids"].to(DEVICE, dtype=torch.long)
                        mask = data["mask"].to(DEVICE, dtype=torch.long)
                        token_type_ids = data["token_type_ids"].to(
                            DEVICE, dtype=torch.long
                        )

                        outputs = self.model(ids, mask, token_type_ids)

                        fin_outputs.extend(
                            torch.sigmoid(outputs).cpu().detach().numpy().tolist()
                        )
                    return fin_outputs

            outputs = run_test()
            outputs = np.array(outputs) >= self.threshold  # threshold
            logger.info("test is done")

            return outputs


if __name__ == "__main__":
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
