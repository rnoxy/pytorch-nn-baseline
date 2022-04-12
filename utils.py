import torch
import torchvision.datasets
from torch.utils.data import DataLoader

from models import MLPNet, ConvNet


def set_seed(seed):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)
    # see https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility


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


def validate_model(model, loss_fn, dataloader: DataLoader):
    # Switch model to eval mode
    model_training = model.training
    model.eval()

    # Do all calculations without accumulating gradients
    running_loss = 0.0
    running_accuracy = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images, labels = batch
            images = images.to(model.device)
            labels = labels.to(model.device)

            ypred = model(images)
            loss = loss_fn(ypred, labels)

            running_loss += loss.item()
            running_accuracy += torch.sum(ypred.argmax(dim=1) == labels)

    # Switch model to training mode
    if model_training:
        model.train()

    running_loss /= batch_idx + 1

    return running_loss, running_accuracy / len(dataloader.dataset)


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