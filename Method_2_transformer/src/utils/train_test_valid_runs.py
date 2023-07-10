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
from src.utils.utils_2 import MultiLabelDataset, DistilBERTClass

from sklearn import metrics
import warnings
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def run_train(epoch):
    model.train()
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
