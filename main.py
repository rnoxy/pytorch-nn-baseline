import torch

from utils import set_seed, validate_model, create_datasets, create_dataloaders, create_model


def main():
    set_seed(seed=42)

    # 1. Create train and valid datasets
    datasets = create_datasets("CIFAR10")

    # 2. Create dataloaders
    batch_size = 20
    dataloaders = create_dataloaders(datasets, batch_size=batch_size)

    # 3. Create model
    model_name = "conv"  # mlp or conv
    model = create_model(model_name)

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
        print(f"Training epoch {epoch + 1} ...")
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
        valid_loss, valid_accuracy = validate_model(
            model, loss_fn, dataloaders["valid"]
        )
        print(
            f"Epoch {epoch + 1}: valid_loss: {valid_loss:.4f}, valid_acc: {valid_accuracy * 100.0: .2f}%"
        )

    torch.save(model.state_dict(), f"{model_name}-epoch{max_epochs}.ckpt")
    print("All done")


if __name__ == "__main__":
    main()
