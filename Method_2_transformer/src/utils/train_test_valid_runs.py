import torch
from tqdm import tqdm

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class RunTrainValidTest:
    def __init__(self, model, loader, weights=None, optimizer=None):
        self.model = model
        self.loader = loader
        self.weights = weights
        self.optimizer = optimizer

    def run_train(self, epoch) -> tuple[list, list]:
        self.model.train()
        fin_outputs = []
        fin_targets = []
        for _, data in tqdm(enumerate(self.loader, 0)):
            ids = data["ids"].to(DEVICE, dtype=torch.long)
            mask = data["mask"].to(DEVICE, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(DEVICE, dtype=torch.long)
            targets = data["valid_targets"].to(DEVICE, dtype=torch.float)
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
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        return fin_outputs, fin_targets

    def run_validation(self) -> tuple[list, list]:
        fin_outputs = []
        fin_targets = []
        self.model.eval()
        with torch.no_grad():
            for _, data in enumerate(self.loader, 0):
                ids = data["ids"].to(DEVICE, dtype=torch.long)
                mask = data["mask"].to(DEVICE, dtype=torch.long)
                token_type_ids = data["token_type_ids"].to(DEVICE, dtype=torch.long)
                targets = data["valid_targets"].to(DEVICE, dtype=torch.float)
                outputs = self.model(ids, mask, token_type_ids)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(
                    torch.sigmoid(outputs).cpu().detach().numpy().tolist()
                )
            return fin_outputs, fin_targets

    def run_test(self) -> tuple[list, list]:
        self.model.eval()
        fin_outputs = []
        fin_targets = []
        with torch.inference_mode():
            for _, data in enumerate(self.loader, 0):
                ids = data["ids"].to(DEVICE, dtype=torch.long)
                mask = data["mask"].to(DEVICE, dtype=torch.long)
                token_type_ids = data["token_type_ids"].to(DEVICE, dtype=torch.long)
                targets = data["valid_targets"].to(DEVICE, dtype=torch.float)
                outputs = self.model(ids, mask, token_type_ids)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(
                    torch.sigmoid(outputs).cpu().detach().numpy().tolist()
                )
            return fin_outputs, fin_targets
