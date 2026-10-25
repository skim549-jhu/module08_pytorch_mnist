'''
utilities.py
Classes & functions to train a Convolutional Neural Network (ConvNet) on the MNIST data
with PyTorch (Module 7, Lecture B, Section 3: "MNIST Classification with PyTorch & GPU")
using Object Oriented Programming (OOP).

Contents:
    get_args()       : Reads the three Command Line Interface (CLI) arguments with argparse
    ConvNet          : The convolutional neural network (subclass of nn.Module)
    MNISTClassifier  : Prepares the data, model, loss, & optimizer & trains the model
    plot()           : Plots & saves the loss & accuracy curves

Runtime: GPU $$$ (CPU on GitHub Codespaces)
argparse:                   https://docs.python.org/3/library/argparse.html
tensors:                    https://pytorch.org/docs/stable/tensors.html
torch.nn:                   https://pytorch.org/docs/stable/nn.html
torch.nn.Module:            https://pytorch.org/docs/stable/generated/torch.nn.Module.html
torch.utils.data:           https://pytorch.org/docs/stable/data.html
torch.no_grad():            https://pytorch.org/docs/stable/generated/torch.no_grad.html
torch.optim:                https://pytorch.org/docs/stable/optim.html
torchvision:                https://pytorch.org/vision/stable/index.html
torchvision.transforms:     https://pytorch.org/vision/stable/transforms.html
torchvision.datasets:       https://pytorch.org/vision/main/datasets.html
'''

# Import the argparse library
# argparse reads the arguments (flags) typed in the terminal
import argparse

# Import the PyTorch library
# PyTorch is an open-source machine learning library used for applications such as computer vision & natural language processing
import torch

# Import the neural network module from PyTorch
# torch.nn provides classes & functions to build neural networks
import torch.nn as nn

# Import the optimization module from PyTorch
# torch.optim provides various optimization algorithms such as SGD, Adam, etc.
import torch.optim as optim

# Import the torchvision library
# torchvision provides tools to handle image & video data
import torchvision

# Import the transforms module from torchvision
# torchvision.transforms provides common image transformations for data augmentation & preprocessing
import torchvision.transforms as transforms

# Import the matplotlib library for plotting
# matplotlib.pyplot provides functions to create various types of plots & visualizations
import matplotlib.pyplot as plt

# Import the tqdm library for progress bars
# tqdm provides a fast, extensible progress bar for loops & tasks
from tqdm import tqdm


#@title Arguments
def get_args():
    """
    Reads the hyperparameters from the terminal Command Line Interface (CLI).

    Example:
        python run.py -e 5 -b 500 -l 0.0005

    Returns:
        argparse.Namespace: An object holding num_epochs, batch_size, & learning_rate.
    """
    # Create the argument parser with a short description shown by "python run.py -h"
    parser = argparse.ArgumentParser(description='MNIST Classification with PyTorch & GPU')

    # Short flag, long flag, data type, default value, & help message of each hyperparameter
    parser.add_argument('-e', '--num_epochs', type=int, default=2,
                        help='Number of times the entire dataset is passed through the model (default: 2)')
    parser.add_argument('-b', '--batch_size', type=int, default=1000,
                        help='Number of samples per batch to be passed through the model (default: 1000)')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001,
                        help='Step size for parameter updates (default: 0.001)')

    # Parse the arguments typed in the terminal (or use the default values)
    return parser.parse_args()


#@title Convolutional neural network
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # First convolutional layer: 1 input channel (grayscale), 32 output channels, 5x5 kernel
        self.layer1 = nn.Sequential(
            #https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            #https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
            nn.BatchNorm2d(32),  # Batch normalization for faster convergence
            # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
            nn.ReLU(),  # Activation function
            # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
            nn.MaxPool2d(kernel_size=2, stride=2))  # Max pooling layer

        # Second convolutional layer: 32 input channels, 64 output channels, 5x5 kernel
        self.layer2 = nn.Sequential(
            #https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            #https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
            nn.BatchNorm2d(64),  # Batch normalization
            # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
            nn.ReLU(),  # Activation function
            # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
            nn.MaxPool2d(kernel_size=2, stride=2))  # Max pooling layer

        # Fully connected layer: input size 7*7*64, output size 1000
        self.fc1 = nn.Linear(7*7*64, 1000)
        # Fully connected layer: input size 1000, output size 10 (number of classes)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        # Forward pass through the first convolutional layer
        out = self.layer1(x)
        # Forward pass through the second convolutional layer
        out = self.layer2(out)
        # Flatten the output for the fully connected layer
        out = out.reshape(out.size(0), -1) # also look at https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html
        # Forward pass through the first fully connected layer
        out = self.fc1(out)
        # Forward pass through the second fully connected layer
        out = self.fc2(out)
        return out


#@title MNIST classifier
class MNISTClassifier:
    """
    Follows the Basic Workflow in PyTorch for the MNIST data:
    Data Preparation -> Model Definition -> Loss Function -> Optimizer -> Training Loop

    Args:
        args (argparse.Namespace): Holds num_epochs, batch_size, & learning_rate (see get_args()).
    """

    def __init__(self, args):
        # Hyperparameters received from the terminal
        self.num_epochs = args.num_epochs  # Number of times the entire dataset is passed through the model
        self.batch_size = args.batch_size  # Number of samples per batch to be passed through the model
        self.learning_rate = args.learning_rate  # Step size for parameter updates

        # Device configuration (use GPU if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. Data Preparation: MNIST dataset & data loaders
        self.train_loader, self.test_loader = self._load_data()

        # 2. Model Definition: create an instance of the model & move it to the configured device (GPU/CPU)
        self.model = ConvNet().to(self.device)

        # 3. Loss Function: CrossEntropyLoss combines nn.LogSoftmax & nn.NLLLoss in one single class
        self.criterion = nn.CrossEntropyLoss()

        # 4. Optimizer: Adam optimizer with the specified learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    # The leading "_" means this method is for internal use inside the class
    def _load_data(self):
        """
        Downloads the MNIST data & wraps it in DataLoaders.

        Returns:
            tuple: The training DataLoader & the testing DataLoader.
        """
        # Download & load the training dataset, applying a transformation to convert images to PyTorch tensors
        train_dataset = torchvision.datasets.MNIST(root='./data',
                                                   train=True,
                                                   transform=transforms.ToTensor(),
                                                   download=True)

        # Download & load the test dataset, applying a transformation to convert images to PyTorch tensors
        test_dataset = torchvision.datasets.MNIST(root='./data',
                                                  train=False,
                                                  transform=transforms.ToTensor())

        # Data loader
        # DataLoader provides an iterable over the dataset with support for batching, shuffling, & parallel data loading
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True)

        # DataLoader for the test dataset, used for evaluating the model
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=False)
        return train_loader, test_loader

    def track_accuracy(self, loader):
        """
        Computes the accuracy (%) of the model on the data of a DataLoader.
        """
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            correct = 0
            total = 0
            for images, labels in loader:
                images = images.to(self.device)  # Move images to the configured device
                labels = labels.to(self.device)  # Move labels to the configured device
                outputs = self.model(images)  # Forward pass
                _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
                total += labels.size(0)  # Increment total by the number of labels
                correct += (predicted == labels).sum().item()  # Increment correct by the number of correct predictions
            accuracy = 100 * correct / total  # Calculate accuracy
        self.model.train()  # Set the model back to training mode
        return accuracy

    def training(self):
        """
        5. Training Loop: trains the model & tracks the loss & accuracies of every epoch.

        Returns:
            tuple: loss_list (average training loss), acc_train, & acc_test (accuracies in %) of every epoch.
        """
        print(f'Device: {self.device} | Epochs: {self.num_epochs} | '
              f'Batch size: {self.batch_size} | Learning rate: {self.learning_rate}')

        # Train the model
        loss_list = []  # List to store loss values
        acc_train = []  # List to store training accuracy values
        acc_test = []  # List to store testing accuracy values

        # Loop over the number of epochs
        for epoch in range(self.num_epochs):
            # Loop over each batch in the training DataLoader
            pbar = tqdm(self.train_loader,
                        desc = f"Training: {epoch + 1:03d}/{self.num_epochs}",
                        ncols = 125,
                        leave = True)

            # Creating running loss empty list
            running_loss = []

            for i, (images, labels) in enumerate(pbar, 1):
                # Important note: In PyTorch, the images in a batch is typically represented as (batch_size, channels, height, width)
                # For example, (100, 1, 28, 28) for the MNIST data
                images = images.to(self.device)  # Move images to the configured device
                labels = labels.to(self.device)  # Move labels to the configured device

                # Forward pass: compute predicted y by passing x to the model
                outputs = self.model(images)
                # Compute the loss
                loss = self.criterion(outputs, labels)
                running_loss.append(loss.item())

                # Backward pass & optimize
                self.optimizer.zero_grad()  # Zero the gradients
                loss.backward()  # Backward pass
                self.optimizer.step()  # Update weights

                # Add the running loss
                pbar.set_postfix(loss = f"{sum(running_loss)/i : 10.6f}")

            # Get the average of all losses in the running_loss as the loss of the current epoch
            loss_list.append(sum(running_loss)/i)

            # Track accuracy on the train set
            acc_train.append(self.track_accuracy(self.train_loader))

            # Track accuracy on the test set
            acc_test.append(self.track_accuracy(self.test_loader))

            # Print the results of the current epoch in the terminal
            print(f'Epoch {epoch + 1:03d}/{self.num_epochs} | Loss: {loss_list[-1]:.6f} | '
                  f'Train Accuracy: {acc_train[-1]:.2f}% | Test Accuracy: {acc_test[-1]:.2f}%')

        return loss_list, acc_train, acc_test


#@title Plot
def plot(loss_list, acc_train, acc_test, save_path='./figure.png'):
    # Plot the loss & accuracy curves
    plt.figure(figsize=(10, 4))

    # Plot the training loss over iterations
    plt.subplot(1, 2, 1)
    plt.plot(loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f"Training Loss")

    # Plot the train & test accuracies over epochs
    plt.subplot(1, 2, 2)
    plt.plot(acc_train)
    plt.plot(acc_test)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f"Model Accuracy")
    plt.legend(['Training', 'Testing'])

    # Save the figure automatically as a .png file (instead of plt.show(), there is no screen in the terminal)
    plt.savefig(save_path)
    plt.close()
    print(f'Figure saved to {save_path}')
