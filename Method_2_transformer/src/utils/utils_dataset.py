import torch
from torch.utils.data import Dataset


class MultiLabelDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, new_data=False):
        """
        Initializes a custom PyTorch Dataset for multi-label text classification.

        Args:
            dataframe (pandas.DataFrame): The input dataframe containing text data and labels.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to tokenize the text.
            max_len (int): The maximum length of the input text after tokenization.
            new_data (bool, optional): Whether the dataset contains new data without labels.
                                       Default is False, which assumes labels are available in the dataframe.
        """
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.text = dataframe.comment_text
        self.new_data = new_data
        if not new_data:
            self.targets = self.dataframe.labels
        self.max_len = max_len

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.text)

    def __getitem__(self, index):
        """
        Gets a single sample from the dataset.

        Parameters:
            index (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the input IDs, attention mask, token type IDs, and valid_targets (if available).
        """
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
            out["valid_targets"] = torch.tensor(self.targets[index], dtype=torch.float)

        return out
