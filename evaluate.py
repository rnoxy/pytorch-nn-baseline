# Epoch 17: valid_loss: 1.2059, valid_acc:  57.40%
# Training epoch 18 ...
#   batch 1000 train_loss: 1.15593
#   batch 2000 train_loss: 1.15194
# Epoch 18: valid_loss: 1.1909, valid_acc:  57.83%
# Training epoch 19 ...
#   batch 1000 train_loss: 1.14728
#   batch 2000 train_loss: 1.13776
# Epoch 19: valid_loss: 1.1709, valid_acc:  58.35%
# Training epoch 20 ...
#   batch 1000 train_loss: 1.12333
#   batch 2000 train_loss: 1.11833
# Epoch 20: valid_loss: 1.2040, valid_acc:  57.80%

import torch
from torch.nn import CrossEntropyLoss

from utils import validate_model, create_datasets, create_dataloaders, create_model


def main(model_path):
    batch_size = 100
    datasets = create_datasets("CIFAR10")
    dataloaders = create_dataloaders(datasets, batch_size=batch_size)
    model = create_model("conv")

    # Load model weights
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    loss_train, acc_train = validate_model(
        model, CrossEntropyLoss(), dataloaders["train"]
    )
    loss_valid, acc_valid = validate_model(
        model, CrossEntropyLoss(), dataloaders["valid"]
    )

    print(f"Train: loss = {loss_train:.4f}, accuracy = {acc_train*100.0: .2f}%")
    print(f"Valid: loss = {loss_valid:.4f}, accuracy = {acc_valid*100.0: .2f}%")


if __name__ == "__main__":
    main("model-epoch20.ckpt")
