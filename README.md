# NN-baseline
Simple baseline project for training _neural network_ 
with clean pytorch code.

In this project we experiment with train the model on _CIFAR10_.
The code structure is as follows:
1. Create datasets.
2. Create dataloaders.
3. Create the model.
4. Create the loss function and optimizer.
5. Run the training loop.

One can simply change the default setup for each element and use your own custom datasets, dataloaders, 
optimizers, etc.

The training loop is very simple in order to show the proper
way of the training models using PyTorch framework.
The most important steps are as follows:
1. Use batches from dataloaders.
2. Remember to call `model.zero_grad()` before calculating the loss function value.
3. Use _backward propagation to calculate the gradients
4. Optimize the model weights (and biases) using `optimizer.step()`.

Below is the sample output for the training of convolutional neural network model. 

```text
Training epoch 1 ...
  batch 1000 train_loss: 2.13389
  batch 2000 train_loss: 1.89950
Epoch 1: valid_loss: 1.8052
Training epoch 2 ...
  batch 1000 train_loss: 1.77746
  batch 2000 train_loss: 1.72301
Epoch 2: valid_loss: 1.6709
Training epoch 3 ...
  batch 1000 train_loss: 1.64631
  batch 2000 train_loss: 1.59872
Epoch 3: valid_loss: 1.5806
```

*Remark.*
In this project we simply load the data using `torchvision` package.
Namely, we used raw `CIFAR10` dataset.
One should **always** look into this data and try to preprocess it.
We skipped that step in order to present only the proper training  
in the PyTorch framework.

Please do not use this code directly to your custom datasets.
You should first do the proper data processing and model weights initialization,
as well as fit the model input shape to your custom dataset images.

