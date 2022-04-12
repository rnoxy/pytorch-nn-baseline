# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torchvision.datasets
from icecream import ic
from torch.utils.data import DataLoader

from models import MLPNet, ConvNet


# Press the green button in the gutter to run the script.
def get_CIFAR10_datasets():
    from torchvision.transforms import Compose, ToTensor

    transform = Compose([ToTensor()])
    train = torchvision.datasets.CIFAR10(
        root="~/datasets", train=True, transform=transform, download=False
    )
    valid = torchvision.datasets.CIFAR10(
        root="~/datasets", train=False, transform=transform, download=False
    )
    return {"train": train, "valid": valid}


def create_datasets(dataset_name: str):
    if dataset_name == "CIFAR10":
        return get_CIFAR10_datasets()
    raise ValueError(f"Unknown dataset name {dataset_name}")


def create_dataloaders(datasets, batch_size=32):

    shuffle = {"train": True, "valid": False}

    return {
        split: DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle[split]
        )
        for split, dataset in datasets.items()
    }


def create_model(model_name: str):
    model_name = model_name.lower()
    if model_name == "mlp":
        return MLPNet()
    elif model_name == "conv":
        return ConvNet()
    else:
        raise ValueError(f"Unknown model name {model_name}.")


def validate_model(model, loss_fn, dataloader: DataLoader):
    # Switch model to eval mode
    model_training = model.training
    model.eval()

    # Do all calculations without accumulating gradients
    running_loss = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images, labels = batch
            images = images.to(model.device)
            labels = labels.to(model.device)

            ypred = model(images)
            loss = loss_fn(ypred, labels)
            # ic(loss.item())
            running_loss += loss.item()

    # Switch model to training mode
    if model_training:
        model.train()

    running_loss /= batch_idx + 1
    return running_loss


def set_seed(seed):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)
    # see https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility


def main():
    set_seed(seed=42)

    # 1. Create train and valid datasets
    datasets = create_datasets("CIFAR10")

    # 2. Create dataloaders
    batch_size = 20
    dataloaders = create_dataloaders(datasets, batch_size=batch_size)

    # 3. Create model
    model = create_model(model_name="conv")  # mlp or conv

    # 3.1. Move all the model weights to the proper device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    ################
    # 4. Training  #
    ################
    max_epochs = 20

    # 4.1 Optimizer
    learning_rate = 0.0001

    from torch.optim import Adam

    optimizer = Adam(params=model.parameters(), lr=learning_rate)

    # 4.2 Loss
    from torch.nn import CrossEntropyLoss

    loss_fn = CrossEntropyLoss()

    for epoch in range(max_epochs):
        print(f"Training epoch {epoch+1} ...")
        running_loss = 0.0
        # for i, batch in tqdm(
        #     enumerate(dataloaders["train"]), total=len(datasets["train"]) // batch_size
        # ):
        for i, batch in enumerate(dataloaders["train"]):
            images, ytrue = batch
            images = images.to(device)
            ytrue = ytrue.to(device)

            ypred = model(images)
            loss = loss_fn(ypred, ytrue)

            # SGD step
            # Zero the gradients before running the backward pass.
            model.zero_grad()

            # Backward pass: compute gradient of the loss with respect to all the learnable
            # parameters of the model. Internally, the parameters of each Module are stored
            # in Tensors with requires_grad=True, so this call will compute gradients for
            # all learnable parameters in the model.
            loss.backward()

            # Do the step of optimizer
            optimizer.step()

            # Gather data and report
            running_loss += (
                loss.item()
            )  # Remark .item() is strongly recommended since we do not want
            # to calculate gradients of `running_loss`

            if i % 1000 == 999:
                last_loss = running_loss / 1000  # loss per batch
                print("  batch {} train_loss: {:.5f}".format(i + 1, last_loss))
                running_loss = 0.0

        # Validate the model
        valid_loss = validate_model(model, loss_fn, dataloaders["valid"])
        print(f"Epoch {epoch+1}: valid_loss: {valid_loss:.4f}")

    # torch.save(model.state_dict(), "model-epoch20.ckpt")
    print("All done")


if __name__ == "__main__":
    main()
