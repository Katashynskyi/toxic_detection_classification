import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel


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


class DistilBERTClass(torch.nn.Module):
    def __init__(self,num_classes=6):
        super(DistilBERTClass, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(768, num_classes),
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        out = hidden_state[:, 0]
        out = self.classifier(out)
        return out

