'''
run.py
Trains a ConvNet on the MNIST data with PyTorch & saves the loss & accuracy plot.

Usage (in the terminal):
    python run.py                          # default values: -e 2 -b 1000 -l 0.001
    python run.py -e 3 -b 500 -l 0.0005    # short flags
    python run.py --num_epochs 3 --batch_size 500 --learning_rate 0.0005  # long flags
    python run.py -h                       # help message
'''

from utilities import get_args, MNISTClassifier, plot

if __name__ == '__main__':
    # Read num_epochs, batch_size, & learning_rate from the terminal
    args = get_args()

    # Create the classifier object (data, model, loss function, & optimizer)
    classifier = MNISTClassifier(args)

    # Print the model for observation
    print(classifier.model)

    # Train the model
    loss_list, acc_train, acc_test = classifier.training()

    # Make some plots!
    plot(loss_list, acc_train, acc_test, save_path='./figure.png')
