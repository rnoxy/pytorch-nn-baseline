import torch
from torch.nn import CrossEntropyLoss

from main import create_datasets, create_dataloaders, create_model, validate_model


def main(model_path):
    batch_size = 100
    datasets = create_datasets("CIFAR10")
    dataloaders = create_dataloaders(datasets, batch_size=batch_size)
    model = create_model("conv")

    # Load model weights
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    loss_train = validate_model(model, CrossEntropyLoss(), dataloaders["train"])
    loss_valid = validate_model(model, CrossEntropyLoss(), dataloaders["valid"])

    print(f"Loss (train) = {loss_train}")
    print(f"Loss (valid) = {loss_valid}")


if __name__ == "__main__":
    main("model-epoch20.ckpt")

    """
    Training epoch 17 ...
      batch 1000 loss: 1.1702054750025273
      batch 2000 loss: 1.168515976548195
    Validation loss = 1.2100261583328247
    
    Training epoch 18 ...
      batch 1000 loss: 1.1655119830369949
      batch 2000 loss: 1.1532788838744163
    Validation loss = 1.1884164599180222
    
    Training epoch 19 ...
      batch 1000 loss: 1.1415360592007637
      batch 2000 loss: 1.138302970468998
    Validation loss = 1.2089203246831894
    """
