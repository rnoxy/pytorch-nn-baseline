import torch
from torch import nn

from models.base import ModelWithDevice


class MLPNet(ModelWithDevice):
    def __init__(self) -> None:
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
