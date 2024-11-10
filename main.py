from src.data_loader import get_data_loader
from src.model import GestureCNN
from src.train import train
from src.evaluate import evaluate
import torch
import torch.optim as optim
import os

def main():
    # Hyperparameters
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 20
    target_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    
    # Data loading
    train_loader, val_loader, test_loader = get_data_loader(target_classes, batch_size)

    # Model and optimizer
    model = GestureCNN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create the 'saved_model' directory if it doesn't exist
    os.makedirs('saved_model', exist_ok=True)
    
    # Dynamically generate the model save path
    model_save_path = f"saved_model/model_{learning_rate}_{batch_size}_{num_epochs}.pth"
    
    # Train and evaluate
    train(model, train_loader, val_loader, optimizer, num_epochs, model_save_path)
    
    # Test the model
    test_err, test_loss = evaluate(model, test_loader)
    print(f'Test Accuracy: {test_err}, Test Loss: {test_loss}')

if __name__ == '__main__':
    main()
