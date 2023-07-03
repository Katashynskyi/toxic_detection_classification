import numpy as np
import pandas as pd
import torch
import argparse

# import mlflow
# from mlflow import log_artifacts, log_metric, log_param
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from src.utils.utils import Split, ReadPrepare
from sklearn import metrics
import warnings
from collections import Counter

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
    ):
        self.path = path
        self.n_samples = n_samples
        self.random_state = random_state
        self.max_len = max_len
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.epochs = (epochs,)
        self.learning_rate = learning_rate

    def do(self):
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
        weights = []
        for i in range(6):
            counts = Counter(train_data["labels"].apply(lambda x: x[i]))
            class_samples, rest = sorted(list(map(int, counts.values())))
            weight = total_samples / (num_classes * class_samples)
            weights.append(round(weight, 4))
        weights = torch.tensor([weights]).to(DEVICE)
        # print(weights)

        # Tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased", truncation=True, do_lower_case=True
        )

        class MultiLabelDataset(Dataset):
            def __init__(self, dataframe, tokenizer, max_len, new_data=False):
                self.dataframe = dataframe
                self.tokenizer = tokenizer
                self.text = dataframe.comment_text
                self.new_data = new_data
                if not new_data:
                    self.targets = self.dataframe.labels
                self.max_len = max_len

            def __len__(self):
                return len(self.text)

            def __getitem__(self, index):
                text = str(self.text[index])
                text = " ".join(text.split())
                text = str(text).lower()

                inputs = self.tokenizer.encode_plus(
                    text,
                    None,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    pad_to_max_length=True,
                    return_token_type_ids=True,
                )
                ids = inputs["input_ids"]
                mask = inputs["attention_mask"]
                token_type_ids = inputs["token_type_ids"]

                out = {
                    "ids": torch.tensor(ids, dtype=torch.long),
                    "mask": torch.tensor(mask, dtype=torch.long),
                    "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                }

                if not self.new_data:
                    out["targets"] = torch.tensor(
                        self.targets[index], dtype=torch.float
                    )

                return out

        train_set = MultiLabelDataset(train_data, tokenizer, self.max_len)
        valid_set = MultiLabelDataset(valid_data, tokenizer, self.max_len)
        test_set = MultiLabelDataset(test_data, tokenizer, self.max_len, new_data=True)

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

        training_loader = DataLoader(train_set, **train_params)
        valid_loader = DataLoader(valid_set, **val_params)
        test_loader = DataLoader(test_set, **val_params)

        class DistilBERTClass(torch.nn.Module):
            def __init__(self):
                super(DistilBERTClass, self).__init__()
                self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
                self.classifier = torch.nn.Sequential(
                    torch.nn.Linear(768, 768),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(768, 6),
                )

            def forward(self, input_ids, attention_mask, token_type_ids):
                output_1 = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                hidden_state = output_1[0]
                out = hidden_state[:, 0]
                out = self.classifier(out)
                return out

        model = DistilBERTClass()
        model.to(DEVICE)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=self.learning_rate)

        def train(epoch):
            model.train()

            for _, data in tqdm(enumerate(training_loader, 0)):
                ids = data["ids"].to(DEVICE, dtype=torch.long)
                mask = data["mask"].to(DEVICE, dtype=torch.long)
                token_type_ids = data["token_type_ids"].to(DEVICE, dtype=torch.long)
                targets = data["targets"].to(DEVICE, dtype=torch.float)

                outputs = model(ids, mask, token_type_ids)

                optimizer.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    outputs, targets, weight=weights
                )

                if _ % 5 == 0:
                    print(f"Epoch: {epoch}, Loss:  {loss.item()}")

                loss.backward()
                optimizer.step()

        for epoch in range(self.epochs):
            print("epoch", epoch)
            train(epoch)

        all_test_pred = []

        def test():
            model.eval()
            with torch.inference_mode():
                for _, data in tqdm(enumerate(test_loader, 0)):
                    ids = data["ids"].to(DEVICE, dtype=torch.long)
                    mask = data["mask"].to(DEVICE, dtype=torch.long)
                    token_type_ids = data["token_type_ids"].to(DEVICE, dtype=torch.long)
                    outputs = model(ids, mask, token_type_ids)
                    probas = torch.sigmoid(outputs)

                    all_test_pred.append(probas)

            return probas

        print("start testing")
        probas = test()
        # print("probas",probas)

        def validation():
            model.eval()
            fin_targets = []
            fin_outputs = []
            with torch.no_grad():
                for _, data in enumerate(valid_loader, 0):
                    ids = data["ids"].to(DEVICE, dtype=torch.long)
                    mask = data["mask"].to(DEVICE, dtype=torch.long)
                    token_type_ids = data["token_type_ids"].to(DEVICE, dtype=torch.long)
                    targets = data["targets"].to(DEVICE, dtype=torch.float)
                    outputs = model(ids, mask, token_type_ids)
                    fin_targets.extend(targets.cpu().detach().numpy().tolist())
                    fin_outputs.extend(
                        torch.sigmoid(outputs).cpu().detach().numpy().tolist()
                    )
            return fin_outputs, fin_targets

        outputs, targets = validation()
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        precision = metrics.precision_score(targets, outputs, average="weighted")
        recall = metrics.recall_score(targets, outputs, average="weighted")
        conf_matrix = metrics.multilabel_confusion_matrix(targets, outputs)
        label_columns = [
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        ]
        for i, matrix in enumerate(conf_matrix):
            print(label_columns[i])
            print(matrix)

        f1_score_micro = metrics.f1_score(targets, outputs, average="micro")
        f1_score_macro = metrics.f1_score(targets, outputs, average="macro")
        print(f"Accuracy Score = {accuracy}")
        print(f"precision Score = {precision}")
        print(f"recall Score = {recall}")
        print(f"F1 Score (Micro) = {f1_score_micro}")
        print(f"F1 Score (Macro) = {f1_score_macro}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--type_of_run", help='"train" of "inference"?', default="train"
    )
    parser.add_argument(
        "--path",
        help="Data path",
        # default="D:/Programming/DB's/toxic_db_for_transformert/train.csv",  # Home-PC
        default="D:/Programming/db's/toxicity_kaggle_1/train.csv",  # Work-PC
    )
    parser.add_argument(
        "--random_state", help="Choose seed for random state", default=42
    )
    parser.add_argument(
        # TODO: ???
        "--max_len",
        help="Max lenght of ???",
        default=100  # home_PC
        # default=512 # work_PC
    )
    parser.add_argument("--train_batch_size", help="Train batch size", default=16)
    parser.add_argument("--valid_batch_size", help="Valid batch size", default=16)
    parser.add_argument("--epochs", help="Number of epochs", default=3)
    parser.add_argument("--learning_rate", help="Learning rate", default=1e-05)
    parser.add_argument("--n_samples", help="How many samples to pass?", default=10000)
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
        )

        classifier.do()
