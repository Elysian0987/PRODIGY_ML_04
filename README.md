````markdown
#  PRODIGY_ML_04: Hand Gesture Recognition using CNN

## Project Overview

This project is part of **Task-04** of the internship at **Prodigy InfoTech**, focused on developing a hand gesture recognition system using deep learning. The model is designed to classify static hand gestures captured in images, enabling natural and intuitive human-computer interaction.

The solution leverages a Convolutional Neural Network (CNN) trained on the LeapGestRecog dataset to accurately detect and classify multiple hand gesture categories.

## Highlights

- **Dataset**: LeapGestRecog dataset from Kaggle  
- **Architecture**: Multi-layered CNN with feature extraction and classification blocks  
- **Framework**: PyTorch  
- **Accuracy**: Achieved 98.83% test accuracy  

## Key Features

- Preprocessing of image data using PyTorch's `DataLoader` and `transforms`
- CNN architecture designed and tuned for image classification tasks
- Model training, validation, and evaluation pipeline
- Visualization of training metrics and evaluation results

## Technologies Used

- Python 3
- PyTorch
- torchvision
- matplotlib
- numpy
- scikit-learn

## Dataset

The project uses the [LeapGestRecog](https://www.kaggle.com/datasets/kmader/leapgestrecog) dataset, which consists of:
- 10 classes of hand gestures
- Over 20,000 grayscale gesture images
- Images captured under consistent lighting conditions using a Leap Motion sensor

## Model Architecture

The CNN architecture includes:
- Multiple convolutional and pooling layers for hierarchical feature extraction
- Batch normalization and dropout for regularization
- Fully connected layers for final classification

## Results

| Metric       | Value     |
|--------------|-----------|
| Test Accuracy| 98.83%    |

The high accuracy indicates effective feature learning and generalization by the model.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gesture-recognition-cnn.git
   cd gesture-recognition-cnn
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run training:

   ```bash
   python train.py
   ```

4. Evaluate the model:

   ```bash
   python evaluate.py
   ```

*(Modify file names based on your actual code files.)*

## Learning Outcomes

* Developed and fine-tuned a deep CNN architecture for image classification
* Learned effective preprocessing using PyTorch’s data pipeline tools
* Understood key evaluation metrics and techniques to assess deep learning model performance

## Acknowledgments

This project was developed under the **Prodigy InfoTech Internship Program** as part of the deep learning track. Special thanks to the mentors for guidance and feedback throughout the task.
