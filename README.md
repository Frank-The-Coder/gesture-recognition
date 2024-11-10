# Gesture Recognition Model

This project implements a **gesture recognition model** using a **Convolutional Neural Network (CNN)** to classify hand gestures representing the first nine letters of the **American Sign Language (ASL)** alphabet (A-I).

The trained model can be used to recognize gestures from hand gesture images and is saved for later use. The model achieves **96.86% accuracy** and a **test loss of 0.1937**.

### **Model Path:**

The model is saved to:

```
saved_model/model_0.001_32_20.pth
```

## Model Details

- **Model Type**: Convolutional Neural Network (CNN)
- **Dataset**: A custom dataset consisting of hand gesture images for ASL letters A to I.
- **Framework**: PyTorch
- **Hyperparameters**:
  - **Learning Rate**: 0.001
  - **Batch Size**: 32
  - **Epochs**: 20

## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/gesture-recognition.git
   cd gesture-recognition
   ```

2. **Set up a Virtual Environment**:
   For managing dependencies, create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   Install the required packages via `pip`:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

The dataset is located in the `data/gestures/Lab3_Gestures_Summer/` directory. The directory structure should be as follows:

```
data/
  └── gestures/
      └── Lab3_Gestures_Summer/
          ├── A/
          ├── B/
          ├── C/
          ├── D/
          ├── E/
          ├── F/
          ├── G/
          ├── H/
          ├── I/
```

Each of the folders `A`, `B`, `C`, etc., contains images for the corresponding ASL letter. If the dataset is in a zip file, it needs to be unzipped into this structure.

![Sample Image](images/sample_images.png)

## Training the Model

You can train the model using the `main.py` script. This will automatically load the dataset, train the model, and save the trained model. Below are the learning curves of the model

![Error Curves](images/error_plot.png)

![Loss Curves](images/loss_plot.png)

### To Train the Model:

1. Modify the `target_classes` in `main.py` (if needed) to adjust the gesture categories (A-I).
2. Run the following command:

   ```bash
   python main.py
   ```

   This will:

   - Train the model for 20 epochs.
   - Save the trained model to `saved_model/model_0.001_32_20.pth`.

### Model Save Path:

After training, the model will be saved in the `saved_model/` directory:

```
saved_model/model_0.001_32_20.pth
```

## Model Evaluation

Once the model is trained and saved, you can evaluate it using the `test.py` script.

### What `test.py` Does:

- **Loading the Model**: The script loads the trained model from the specified path (`saved_model/model_0.001_32_20.pth`).
- **Test Evaluation**: It uses the `evaluate()` function to calculate the **test error** and **test loss** on the test dataset.
- **Results**: After evaluation, the script prints the **Test Accuracy** (calculated as `1 - test error`) and **Test Loss** to give insights into the model's performance.

### CUDA and GPU Usage:

The model is designed to automatically leverage **GPU** acceleration if available, ensuring faster training and evaluation. If a GPU is not available, the model will seamlessly fall back to **CPU**. This functionality enhances performance, particularly during the training phase, enabling the model to handle larger datasets more efficiently.

### To Evaluate the Model:

1. Ensure that your **test_loader** is correctly set up.
2. Run the following command:
   ```bash
   python test.py
   ```

### Test Results:

- **Test Accuracy**: 96.86%
- **Test Loss**: 0.1937

## File Structure

```
gesture-recognition/
│
├── src/
│   ├── data_loader.py       # Code to load and process the dataset
│   ├── model.py             # Model architecture (GestureCNN)
│   ├── train.py             # Training function
│   ├── evaluate.py          # Evaluation function
│   └── plot.py              # Code for plotting training/validation curves
├── data/                    # Dataset directory (containing the gestures dataset)
│   └── gestures/
│       └── Lab3_Gestures_Summer/
│           ├── A/
│           ├── B/
│           ├── C/
│           ├── D/
│           ├── E/
│           ├── F/
│           ├── G/
│           ├── H/
│           ├── I/
├── saved_model/             # Directory where the trained models are saved
├── main.py                  # Main script to train and evaluate the model
├── test.py                  # Script to evaluate the saved model
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Requirements

- **PyTorch**: This project uses PyTorch for model training and evaluation.
- **Matplotlib**: Used for visualizing the training and validation curves.
- **NumPy**: Used for numerical operations and storing statistics.

You can install all dependencies by running:

```bash
pip install -r requirements.txt
```

## Conclusion

This project implements a **gesture recognition model** using a CNN architecture. The model was trained on a custom dataset of hand gestures and achieved a **test accuracy of 96.86%**. The trained model is saved and can be loaded for evaluation or inference.

## Licensing

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
