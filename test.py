import torch
from src.model import GestureCNN  # Your model class
from src.evaluate import evaluate  # Your evaluate function
from src.data_loader import get_data_loader

def load_model(model_path):
    """
    Loads the model from the specified path and returns the model.
    
    Args:
        model_path: Path to the saved model.
        
    Returns:
        model: Loaded model with weights.
    """
    # Reinitialize the model
    model = GestureCNN()

    # Load the state_dict (weights) from the saved model
    model.load_state_dict(torch.load(model_path))
    
    # Set the model to evaluation mode
    model.eval()
    
    return model

def main():
    # Hyperparameters
    batch_size = 32
    target_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    # Model path (where the final model is saved)
    model_save_path = "saved_model/model_0.001_32_20.pth"
    
    # Load the model
    model = load_model(model_save_path)

    _, _, test_loader = get_data_loader(target_classes, batch_size)

    # Evaluate the model on the test set
    # Assuming you have the test_loader defined
    test_err, test_loss = evaluate(model, test_loader)
    print(f'Test Accuracy: {test_err}, Test Loss: {test_loss}')

if __name__ == '__main__':
    main()
