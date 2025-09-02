# Kannada-MNIST Classification (MLP from Scratch)

## Objective
Develop a feedforward neural network (MLP) implemented from scratch (using NumPy) to classify handwritten digits from the Kannada-MNIST dataset, exploring different activation functions and training strategies.

## Dataset
- **Source**: [Kannada MNIST(Kaggle)](https://www.kaggle.com/c/Kannada-MNIST?utm_source=chatgpt.com)
- **Format**: CSV files (train.csv, test.csv) with pixel values (28×28 grayscale)
- **Classes**: 0 digit classes (0–9 in the Kannada script)
- **Preprocessing**:
    - Normalization
    - One-hot encoding of labels
    - Train/test split (80/20)

## Approach
- Data loading and preprocessing using Pandas and NumPy
- Custom implementation of MLP with one hidden layer
- Forward propagation and backward propagation coded manually
- Experiments with multiple activation functions (ReLU, Leaky ReLU, ELU, Tanh)
- Training loop with evaluation on validation/test set
- Tracking train/test loss and accuracy across epochs

## Methods / Algorithms
- **Network Architecture**:
  - Input layer: 784 features (28×28 pixels)
  - Hidden layer: configurable size (e.g., 128 units)
  - Output layer: 10 units with softmax
- **Training**:
  - Gradient descent with manual weight updates
  - Cross-entropy loss
  - Accuracy metric for evaluation
- **Activation Functions Tested**:
  - ReLU
  - Leaky ReLU
  - ELU
  - Tanh

## Evaluation Metrics
- Accuracy
- Cross-Entropy Loss
- Train vs. Test curves for performance monitoring
- Number of Incorrect Predicting Labels

## Results
- **ELU with learning rate = 1.2** showed the best performance:
  - Accuracy of Prediction: 97.75%.
  - Number of Incorrect Predicting Labels: 269

## Tools & Libraries
- **Numerical Computing**: NumPy
- **Data Handling**: Pandas
- **Visualization**: Matplotlib

## Key Insights / Business Value
- Demonstrates how neural networks can be built and trained from scratch without deep learning frameworks.
- Provides educational value in understanding forward and backward propagation, gradient descent, and activation function effects.
- Kannada-MNIST is a useful benchmark for testing models on scripts beyond standard Latin digits.

## ELU with learning rate = 1.2 
- **Metrics**
<Figure size 600x600 with 2 Axes><img width="590" height="590" alt="image" src="https://github.com/user-attachments/assets/60207191-7962-4da2-8474-33272f1dd112" />

  - **Examples of incorrect predictions**
<Figure size 800x600 with 6 Axes><img width="790" height="574" alt="image" src="https://github.com/user-attachments/assets/5390bf3c-0afa-48d9-83a9-91a3ef60b31d" />

