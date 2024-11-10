import torch
import torch.nn as nn
import torch.optim as optim

def train(model, train_loader, val_loader, optimizer, num_epochs, model_save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Move the model to the appropriate device
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Loop through the training batches
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move to device
            optimizer.zero_grad()  # Zero gradients
            
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)
            loss.backward()  # Backward pass
            optimizer.step()  # Update the weights
            
            # Calculate training loss and error
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Calculate training loss and error
        train_loss = running_loss / len(train_loader)
        train_error = 1 - (correct / total)  # Error = 1 - Accuracy

        # Evaluate the model on the validation set
        val_loss, val_error = evaluate(model, val_loader, criterion)

        # Print the stats for this epoch
        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Error: {train_error:.4f} | '
              f'Val Loss: {val_loss:.4f}, Val Error: {val_error:.4f}')

    # Save the final model after training
    torch.save(model.state_dict(), model_save_path)
    print(f"Final model saved to {model_save_path}")


def evaluate(model, val_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # Set to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move to device
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate average validation loss and error
    val_loss /= len(val_loader)
    val_error = 1 - (correct / total)  # Error = 1 - Accuracy
    
    return val_loss, val_error
