from torch import nn


def build_loss(name: str = "cross_entropy") -> nn.Module:
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported loss: {name}")
