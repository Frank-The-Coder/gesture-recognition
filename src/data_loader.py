import zipfile
import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def unzip_dataset(zip_path, extract_to_path):
    """
    Unzips the dataset.zip file if it hasn't been extracted already.
    :param zip_path: Path to the .zip file.
    :param extract_to_path: Path where the dataset should be extracted.
    """
    if not os.path.exists(extract_to_path):
        print(f"Unzipping the dataset from {zip_path} to {extract_to_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_path)
        print(f"Dataset extracted to {extract_to_path}")
    else:
        print(f"The dataset is already extracted at {extract_to_path}.")

def get_data_loader(target_classes, batch_size):
    # Path to the zip file and extracted folder
    zip_file_path = 'data/dataset.zip'  # Change this if your zip file has a different name
    extracted_data_path = 'data/gestures'  # Path where the dataset will be extracted
    # Unzip dataset if not already extracted
    unzip_dataset(zip_file_path, extracted_data_path)
    data_path = extracted_data_path + '/Lab3_Gestures_Summer'
    
    # Define the image transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to a standard size
        transforms.ToTensor(),  # Convert image to PyTorch tensor
    ])
    
    # Load the dataset using ImageFolder
    dataset = datasets.ImageFolder(data_path, transform=transform)
    
    # Filter dataset for target classes (e.g., A to Z)
    relevant_indices = [i for i, (_, label) in enumerate(dataset) if dataset.classes[label] in target_classes]
    filtered_dataset = torch.utils.data.Subset(dataset, relevant_indices)
    
    # Split the data into train, validation, and test sets (80% train, 10% validation, 10% test)
    train_size = int(0.8 * len(filtered_dataset))
    val_size = int(0.1 * len(filtered_dataset))
    test_size = len(filtered_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(filtered_dataset, [train_size, val_size, test_size])

    # DataLoader for batching the data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
