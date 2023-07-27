import os
import torch
from transformers import DistilBertModel


class DistilBERTClass(torch.nn.Module):
    def __init__(self, num_classes=6):
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

    def _save(
        self, filepath="model/model_weights.pth", epoch=None, model=None, optimizer=None
    ):
        """return 800 mb pth file"""
        # TODO: look at Case # 2: Save model to resume training later
        "https://stackoverflow.com/questions/42703500/how-do-i-save-a-trained-model-in-pytorch"
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
        # torch.save(self.state_dict(), filepath)

    def save(self, filepath="model/model_weights.pth"):
        try:
            os.mkdir("model")
        except FileExistsError:
            pass
        torch.save(self.state_dict(), filepath)

    def load(self, filepath="model/model_weights.pth"):
        # Load the model's state dictionary from the specified filepath
        self.load_state_dict(torch.load(filepath))


if __name__ == "__main__":
    DistilBERTClass().save()
    # DistilBERTClass.save()
