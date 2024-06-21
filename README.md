# Paradigm-programming

Comprised of four sections, including ASP, Generalisation Priority Algorithms in Haskell, Machine Learning in Physical Currents, and Quantum Physics Learning
## ASP
## Generalisation Priority Algorithms in Haskell
## Machine Learning in Physical Currents （Mark ： 86/100 ）

This section contains a Python script that implements a neural network model for regression tasks, specifically designed for analyzing physical current measurements.

### Features

- Data loading and preprocessing from CSV files
- Data visualization using matplotlib
- Custom neural network architecture using PyTorch
- Training process with configurable hyperparameters
- Model evaluation and result visualization

### Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Pandas

### Usage

1. Ensure you have a CSV file named 'measurements.csv' in the same directory as the script.
2. Run the script to train the model and visualize results.

### Model Architecture

- Input layer: 1 neuron
- Hidden layer 1: 100 neurons with LeakyReLU activation and Dropout (0.2)
- Hidden layer 2: 100 neurons with LeakyReLU activation and Dropout (0.2)
- Output layer: 1 neuron

### Optimization

- Loss function: Mean Squared Error (MSE)
- Optimizer: RMSprop with learning rate of 0.001 and weight decay of 0.001
- Learning rate scheduler: StepLR with step size of 400 and gamma of 0.5

### Results

The script will output:
1. Training progress every 100 epochs
2. Scatter plots of actual vs. predicted values for both training and test sets

### Note

This implementation is part of a larger project exploring various programming paradigms and their applications in scientific computing and physics.
