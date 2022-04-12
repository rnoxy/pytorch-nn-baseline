import torch
import torchvision.datasets
from torch.utils.data import DataLoader


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
