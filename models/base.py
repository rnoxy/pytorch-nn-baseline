import torch


class ModelWithDevice(torch.nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device
