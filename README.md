# Neural Net From Scratch

This project demonstrates how to build a simple feedforward neural network from scratch in Python, without using machine learning libraries like TensorFlow or PyTorch. The network is trained to perform binary classification on a tabular dataset.

## Features
- Data loading and preprocessing (CSV format)
- Train/test split
- Feature scaling (Min-Max normalization)
- Manual implementation of matrix multiplication and bias addition
- Forward propagation using the sigmoid activation function
- Backpropagation and gradient descent for training
- Saving and loading model weights and biases
- Model evaluation on test data

## Project Structure
```
Neural_Net.ipynb        # Jupyter notebook with code and explanations
Neural_Net.py           # Python script version (if available)
data/
  full_data.csv         # Full dataset
  train_data.csv        # Training set
  test_data.csv         # Test set
model_weights/
  W1.csv, b1.csv        # Weights and biases for hidden layer
  W2.csv, b2.csv        # Weights and biases for output layer
README.md               # Project documentation
```

## How to Run
1. **Prepare the data:** Place your CSV data in the `data/` folder. The code expects columns for features (e.g., income, age, loan) and a target class.
2. **Run the notebook:** Open `Neural_Net.ipynb` in Jupyter and execute the cells step by step. Follow the markdown explanations for guidance.
3. **Train the model:** The training loop will update weights and biases using backpropagation.
4. **Evaluate:** After training, the model's accuracy on the test set will be displayed.
5. **Save/Load weights:** Model parameters are saved in the `model_weights/` folder for future use.

## Requirements
- Python 3.x
- Jupyter Notebook (for `.ipynb`)
- No external ML libraries required

## Learning Goals
- Understand the inner workings of neural networks
- Practice implementing core ML concepts from scratch
- Gain experience with data preprocessing and model evaluation

## References
- [Neural network (Wikipedia)](https://en.wikipedia.org/wiki/Artificial_neural_network)
- [Backpropagation (Wikipedia)](https://en.wikipedia.org/wiki/Backpropagation)

---
Feel free to modify and experiment with the code to deepen your understanding of neural networks!
