import matplotlib.pyplot as plt

def plot_training_curve(train_loss, val_loss, train_acc, val_acc, save_path=None):
    """
    Plots the training and validation loss and accuracy over epochs.

    :param train_loss: List or array of training loss values over epochs
    :param val_loss: List or array of validation loss values over epochs
    :param train_acc: List or array of training accuracy values over epochs
    :param val_acc: List or array of validation accuracy values over epochs
    :param save_path: Path to save the plot image, if provided
    """
    epochs = range(1, len(train_loss) + 1)  # Assuming loss and accuracy lists are of same length

    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)  # Create subplots
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
