#!/usr/bin/env python
# coding: utf-8

# # Fine Tuning DistilBERT for Toxic Comment Classification Challenge

# # Introduction
#
# I will be fine tuning a DistilBert transformer model for the **Toxic Comment MultiLabel Classification** problem.
# This is one of the most common business problems where a given piece of text/sentence/document needs to be classified into one or more of categories out of the given list.
#
#
# <center><img src="https://www.mathworks.com/help/examples/nnet/win64/MultilabelImageClassificationUsingDeepLearningExample_01.png" alt="Multi Label example" title="MultiLabel classification example" width="70%"   /></center>
#
#
# ## Table of Contents
#
# 1. [Importing Python Libraries](#section01)
# 2. [Pre-Processing the domain data](#section02)
# 3. [Training Parameters](#section03)
# 4. [Preparing the Dataset and Dataloader](#section04)
# 4. [Neural Network for Fine Tuning](#section05)
# 5. [Fine Tuning the Model](#section06)
# 6. [Generate Submissions.csv](#section07)
#
# ## Technical Details
#
# This script leverages on multiple tools designed by other teams. Details of the tools used below. Please ensure that these elements are present in your setup to successfully implement this script.
#
#  - Data:
# 	 - We are referring only to the first csv file from the data dump: `train.csv`
# 	 - There are rows of data.  Where each row has the following data-point:
# 		 - Comment Text
# 		 - `toxic`
# 		 - `severe_toxic`
# 		 - `obscene`
# 		 - `threat`
# 		 - `insult`
# 		 - `identity_hate`
#
# Each comment can be marked for multiple categories. If the comment is `toxic` and `obscene`, then for both those headers the value will be `1` and for the others it will be `0`.
#
#
#  - Language Model Used:
# 	 - DistilBERT is a smaller transformer model as compared to BERT or Roberta. It is created by process of distillation applied to Bert.
# 	 - [Blog-Post](https://medium.com/huggingface/distilbert-8cf3380435b5)
# 	 - [Research Paper](https://arxiv.org/pdf/1910.01108)
#      - [Documentation for python](https://huggingface.co/transformers/model_doc/distilbert.html)
#


# ***NOTE***
# - *It is to be noted that the overall mechanisms for a multiclass and multilabel problems are similar, except for few differences namely:*
# 	- *Loss function is designed to evaluate all the probability of categories individually rather than as compared to other categories. Hence the use of `BCE` rather than `Cross Entropy` when defining loss.*
# 	- *Sigmoid of the outputs calcuated to rather than Softmax. Again for the reasons defined in the previous point*

# # Importing Python Libraries <a id='section01'></a>
#
# At this step we will be importing the libraries and modules needed to run our script. Libraries are:

# In[1]:


import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Setting up the device for GPU usage
#
# Followed by that we will preapre the device for CUDA execeution. This configuration is needed if you want to leverage on onboard GPU.

# In[2]:


from torch import cuda

device = torch.device("cuda" if cuda.is_available() else "cpu")

print(f"Current device: {device}")


# # Pre-Processing the domain data <a id='section02'></a>
#
# We will be working with the data and preparing for fine tuning purposes.
# *Assuming that the `train.csv` is already downloaded, unzipped and saved in your `data` folder*
#
# * First step will be to remove the **id** column from the data.
# * The values of all the categories and coverting it into a list.
# * The list is appened as a new column names as **labels**.
# * Drop all the labels columns

# In[3]:

path_train = "D:/Programming/db's/toxicity_kaggle_2/train.csv"

train_data = pd.read_csv(path_train)
# print(f"Total Training Records : {len(train_data)}")

train_data.head()


# ## Removing id column and preparing labels into the single list column

# In[4]:
train_size = 0.001
train_data = train_data.sample(frac=train_size, random_state=123).reset_index(drop=True)
train_data.drop(["id"], inplace=True, axis=1)
train_data["labels"] = train_data.iloc[:, 1:].values.tolist()
train_data.drop(train_data.columns.values[1:-1].tolist(), inplace=True, axis=1)
print(train_data.head())


# ## Data Cleaning
#
# - Lower case
# - Remove extra space

# In[5]:


train_data["comment_text"] = train_data["comment_text"].str.lower()
train_data["comment_text"] = (
    train_data["comment_text"]
    .str.replace("\xa0", " ", regex=False)
    .str.split()
    .str.join(" ")
)


# # Training Parameters <a id='section03'></a>
#
# Defining some key variables that will be used later on in the training
#

# In[6]:


MAX_LEN = 512
TRAIN_BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 1e-05
NUM_WORKERS = 2


# # Preparing the Dataset and Dataloader <a id='section04'></a>
# We will start with defining few key variables that will be used later during the training/fine tuning stage.
# Followed by creation of MultiLabelDataset class - This defines how the text is pre-processed before sending it to the neural network. We will also define the Dataloader that will feed  the data in batches to the neural network for suitable training and processing.
# Dataset and Dataloader are constructs of the PyTorch library for defining and controlling the data pre-processing and its passage to neural network. For further reading into Dataset and Dataloader read the [docs at PyTorch](https://pytorch.org/docs/stable/data.html)
#
# ## *MultiLabelDataset* Dataset Class
# - This class is defined to accept the `tokenizer`, `dataframe`, `max_length` and `eval_mode` as input and generate tokenized output and tags that is used by the BERT model for training.
# - We are using the DistilBERT tokenizer to tokenize the data in the `text` column of the dataframe.
# - The tokenizer uses the `encode_plus` method to perform tokenization and generate the necessary outputs, namely: `ids`, `attention_mask`, `token_type_ids`
#
# - To read further into the tokenizer, [refer to this document](https://huggingface.co/transformers/model_doc/distilbert.html#distilberttokenizer)
# - `targets` is the list of categories labled as `0` or `1` in the dataframe.
# - The *MultiLabelDataset* class is used to create 2 datasets, for training and for validation.
# - *Training Dataset* is used to fine tune the model: **80% of the original data**
# - *Validation Dataset* is used to evaluate the performance of the model. The model has not seen this data during training.
#
# ## Dataloader
# - Dataloader is used to for creating training and validation dataloader that load data to the neural network in a defined manner. This is needed because all the data from the dataset cannot be loaded to the memory at once, hence the amount of dataloaded to the memory and then passed to the neural network needs to be controlled.
# - This control is achieved using the parameters such as `batch_size` and `max_len`.
# - Training and Validation dataloaders are used in the training and validation part of the flow respectively

# In[7]:


class MultiLabelDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len: int, eval_mode: bool = False):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.text = dataframe.comment_text
        self.eval_mode = eval_mode
        if self.eval_mode is False:
            self.targets = self.data.labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text.iloc[index])
        text = " ".join(text.split())

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

        output = {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        }

        if self.eval_mode is False:
            output["targets"] = torch.tensor(
                self.targets.iloc[index], dtype=torch.float
            )

        return output


# ## Loading tokenizer and generating training set

# In[8]:


tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased", truncation=True, do_lower_case=True
)
training_set = MultiLabelDataset(train_data, tokenizer, MAX_LEN)


# ## Verify the data at index 0

# In[9]:


training_set[0]


# ## Creating Dataloader

# In[10]:


train_params = {
    "batch_size": TRAIN_BATCH_SIZE,
    "shuffle": True,
    #'num_workers': NUM_WORKERS
}
training_loader = DataLoader(training_set, **train_params)


# <a id='section05'></a>
# # Neural Network for Fine Tuning
#
# ## Neural Network
#  - We will be creating a neural network with the `DistilBERTClass`.
#  - This network will have the `DistilBERT` model.  Follwed by a `Droput` and `Linear Layer`. They are added for the purpose of **Regulariaztion** and **Classification** respectively.
#  - In the forward loop, there are 2 output from the `DistilBERTClass` layer.
#  - The second output `output_1` or called the `pooled output` is passed to the `Drop Out layer` and the subsequent output is given to the `Linear layer`.
#  - Keep note the number of dimensions for `Linear Layer` is **6** because that is the total number of categories in which we are looking to classify our model.
#  - The data will be fed to the `DistilBERTClass` as defined in the dataset.
#  - Final layer outputs is what will be used to calcuate the loss and to determine the accuracy of models prediction.
#  - We will initiate an instance of the network called `model`. This instance will be used for training and then to save the final trained model for future inference.

# In[11]:


# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model.


class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 6)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


# ## Loading Neural Network model

# In[12]:


model = DistilBERTClass()
model.to(device)


# ## Loss Function and Optimizer
#  - The Loss is defined in the next cell as `loss_fn`.
#  - As defined above, the loss function used will be a combination of Binary Cross Entropy which is implemented as [BCELogits Loss](https://pytorch.org/docs/stable/nn.html#bcewithlogitsloss) in PyTorch
#  - `Optimizer` is defined in the next cell.
#  - `Optimizer` is used to update the weights of the neural network to improve its performance.

# In[13]:


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


# In[14]:


optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


# <a id='section06'></a>
# # Fine Tuning the Model
#
# After all the effort of loading and preparing the data and datasets, creating the model and defining its loss and optimizer. This is probably the easier steps in the process.
#
# Here we define a training function that trains the model on the training dataset created above, specified number of times (EPOCH), An epoch defines how many times the complete data will be passed through the network.
#
# Following events happen in this function to fine tune the neural network:
# - The dataloader passes data to the model based on the batch size.
# - Subsequent output from the model and the actual category are compared to calculate the loss.
# - Loss value is used to optimize the weights of the neurons in the network.
# - After every 5000 steps the loss value is printed in the console.
#
# As you can see just in 1 epoch by the final step the model was working with a miniscule loss of 0.05 i.e. the network output is extremely close to the actual output.

# In[15]:


def train(epoch):
    model.train()
    for _, data in tqdm(enumerate(training_loader, 0)):
        ids = data["ids"].to(device, dtype=torch.long)
        mask = data["mask"].to(device, dtype=torch.long)
        token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
        targets = data["targets"].to(device, dtype=torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _ % 50 == 0:
            print(f"Epoch: {epoch}, Loss:  {loss.item()}")

        loss.backward()
        optimizer.step()


# In[16]:


for epoch in range(EPOCHS):
    train(epoch)


# # Generate Submissions.csv <a id='section07'></a>

# In[17]:

path_test = "D:/Programming/db's/toxicity_kaggle_2/test.csv"
test_data = pd.read_csv(path_test)
test_size = 0.001
train_data = test_data.sample(frac=test_size, random_state=123).reset_index(drop=True)
# print(test_data.head())


# In[18]:


test_set = MultiLabelDataset(test_data, tokenizer, MAX_LEN, eval_mode=True)
testing_params = {
    "batch_size": TRAIN_BATCH_SIZE,
    "shuffle": False,
    #'num_workers': 2
}
test_loader = DataLoader(test_set, **testing_params)


# In[19]:


all_test_pred = []


def test(epoch):
    model.eval()

    with torch.inference_mode():
        for _, data in tqdm(enumerate(test_loader, 0)):
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            outputs = model(ids, mask, token_type_ids)
            probas = torch.sigmoid(outputs)

            all_test_pred.append(probas)
    return probas


# In[20]:


probas = test(model)


# In[21]:


all_test_pred = torch.cat(all_test_pred)


# In[22]:


submit_df = test_data.copy()
submit_df.drop("comment_text", inplace=True, axis=1)


# In[23]:


label_columns = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


# In[24]:


for i, name in enumerate(label_columns):
    submit_df[name] = all_test_pred[:, i].cpu()
    submit_df.head()


# In[25]:


submit_df.to_csv("submission.csv", index=False)
# submit_df.head()


# In[ ]:
