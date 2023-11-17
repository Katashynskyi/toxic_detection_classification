import os
import torch
from transformers import DistilBertModel


class DistilBERTClass(torch.nn.Module):
    """
    A PyTorch module for fine-tuning DistilBERT for our classification tasks.
    """

    def __init__(self, num_classes=6):
        """
        Initialize the DistilBERT classifier.

        Args:
            num_classes (int, optional): The number of output classes. Default is 6.
        """
        super(DistilBERTClass, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(768, num_classes),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass through the DistilBERT model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (torch.Tensor): Token type IDs.

        Returns:
            torch.Tensor: Model predictions.
        """
        output_1 = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        out = hidden_state[:, 0]
        out = self.classifier(out)
        return out

    @staticmethod
    def _save(
        filepath="model/model_weights.pth", epoch=None, model=None, optimizer=None
    ):
        """
        Save the model's state dictionary and other relevant data.

        Args:
            filepath (str, optional): The file path to save the model. Default is "model/model_weights.pth".
            epoch (int, optional): The current epoch. Default is None.
            model (nn.Module, optional): The model to save. Default is None.
            optimizer (Optimizer, optional): The optimizer state to save. Default is None.
        """
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        try:
            os.mkdir("../../model/")
        except:
            pass

        torch.save(state, filepath)

    def save(self, filepath="model/model_weights.pth"):
        """
        Save the model's state dictionary.

        Args:
            filepath (str, optional): The file path to save the model. Default is "model/model_weights.pth".
        """
        try:
            os.mkdir("model")
        except FileExistsError:
            pass
        torch.save(self.state_dict(), filepath)

    def load(self, filepath="model/model_weights1e-05.pth"):
        """
        Load the model's state dictionary from the specified filepath.

        Args:
            filepath (str, optional): The file path to load the model from. Default is "model/model_weights1e-05.pth".
        """
        self.load_state_dict(torch.load(filepath))
        return self
