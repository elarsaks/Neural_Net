import csv
import math
import random

# Read CSV file without checking for missing values
data = []
with open("data/full_data.csv", newline="") as csvfile:
    reader = csv.reader(csvfile)
    headers = next(reader)  # Skip the header
    for row in reader:
        data.append(row)


# Shuffle the dataset
random.seed(42)  # For reproducibility
random.shuffle(data)


# Define the split ratio (80% training, 20% testing)
split_ratio = 0.8
split_index = int(len(data) * split_ratio)


# Split the data into training and testing sets
train_data = data[:split_index]
test_data = data[split_index:]


# Save the training data to a CSV file
with open("data/train_data.csv", "w", newline="") as trainfile:
    writer = csv.writer(trainfile)
    writer.writerow(headers)  # Write the header
    writer.writerows(train_data)  # Write the training data


# Save the testing data to a CSV file
with open("data/test_data.csv", "w", newline="") as testfile:
    writer = csv.writer(testfile)
    writer.writerow(headers)  # Write the header
    writer.writerows(test_data)  # Write the testing data


# Convert string data to numerical values (for relevant columns)
X_train = [
    [float(row[1]), float(row[2]), float(row[3])] for row in train_data
]  # income, age, loan
y_train = [int(row[4]) for row in train_data]  # class (target)

X_test = [
    [float(row[1]), float(row[2]), float(row[3])] for row in test_data
]  # income, age, loan
y_test = [int(row[4]) for row in test_data]  # class (target)


# Manually normalize the data using Min-Max scaling
def min_max_scaling(X):
    min_vals = [min(col) for col in zip(*X)]
    max_vals = [max(col) for col in zip(*X)]

    X_scaled = []
    for row in X:
        scaled_row = [
            (row[i] - min_vals[i]) / (max_vals[i] - min_vals[i])
            for i in range(len(row))
        ]
        X_scaled.append(scaled_row)

    return X_scaled


X_train_scaled = min_max_scaling(X_train)
X_test_scaled = min_max_scaling(X_test)


# Sigmoid function
def sigmoid(z):
    return 1 / (1 + math.exp(-z))


# Derivative of the sigmoid function
def sigmoid_derivative(a):
    return a * (1 - a)


# Function to initialize random weights without using numpy
def random_matrix(rows, cols):
    return [[random.random() for _ in range(cols)] for _ in range(rows)]


# Initialize weights and biases
input_size = len(X_train_scaled[0])  # 3 input features: income, age, loan
hidden_size = 4
output_size = 1


# Randomly initialize weights and biases
W1 = random_matrix(input_size, hidden_size)
b1 = [random.random() for _ in range(hidden_size)]
W2 = random_matrix(hidden_size, output_size)
b2 = [random.random() for _ in range(output_size)]


# Matrix multiplication function (for dot product)
def matrix_multiply(A, B):
    return [
        [sum(a * b for a, b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A
    ]


# Adding bias to a matrix
def add_bias(matrix, bias):
    return [
        [matrix[row][col] + bias[col] for col in range(len(bias))]
        for row in range(len(matrix))
    ]


# Element-wise application of the sigmoid function
def apply_sigmoid(matrix):
    return [[sigmoid(x) for x in row] for row in matrix]


# Forward propagation
def forward(X, W1, b1, W2, b2):
    Z1 = add_bias(matrix_multiply(X, W1), b1)  # Input to hidden layer
    A1 = apply_sigmoid(Z1)  # Activation in hidden layer
    Z2 = add_bias(matrix_multiply(A1, W2), b2)  # Input to output layer
    A2 = apply_sigmoid(Z2)  # Final output (prediction)
    return A1, A2


# Backpropagation (for one epoch)
def backprop(X, y, W1, b1, W2, b2, A1, A2, learning_rate=0.1):
    m = len(y)

    dZ2 = [[A2[i][0] - y[i]] for i in range(m)]  # Derivative of loss w.r.t output
    dW2 = [
        [sum(A1[i][h] * dZ2[i][0] for i in range(m)) / m for _ in range(output_size)]
        for h in range(hidden_size)
    ]
    db2 = [sum(dZ2[i][0] for i in range(m)) / m]

    dA1 = [
        [
            sum(W2[h][o] * dZ2[i][0] for o in range(output_size))
            for h in range(hidden_size)
        ]
        for i in range(m)
    ]
    dZ1 = [
        [dA1[i][h] * sigmoid_derivative(A1[i][h]) for h in range(hidden_size)]
        for i in range(m)
    ]
    dW1 = [
        [sum(X[i][f] * dZ1[i][h] for i in range(m)) / m for h in range(hidden_size)]
        for f in range(input_size)
    ]
    db1 = [sum(dZ1[i][h] for i in range(m)) / m for h in range(hidden_size)]

    # Update weights and biases using gradient descent
    W1 = [
        [W1[f][h] - learning_rate * dW1[f][h] for h in range(hidden_size)]
        for f in range(input_size)
    ]
    b1 = [b1[h] - learning_rate * db1[h] for h in range(hidden_size)]
    W2 = [
        [W2[h][o] - learning_rate * dW2[h][o] for o in range(output_size)]
        for h in range(hidden_size)
    ]
    b2 = [b2[o] - learning_rate * db2[o] for o in range(output_size)]

    return W1, b1, W2, b2


# Training loop
for epoch in range(10000):  # Number of epochs
    A1, A2 = forward(X_train_scaled, W1, b1, W2, b2)
    W1, b1, W2, b2 = backprop(X_train_scaled, y_train, W1, b1, W2, b2, A1, A2)

    # Optional: Calculate loss for monitoring
    if epoch % 1000 == 0:
        loss = sum(
            -y_train[i] * math.log(A2[i][0]) - (1 - y_train[i]) * math.log(1 - A2[i][0])
            for i in range(len(y_train))
        ) / len(y_train)
        print(f"Epoch {epoch}, Loss: {loss}")


# Save the weights and biases after training
def save_weights_biases(W, b, W_file, b_file):
    with open(W_file, "w") as f_w, open(b_file, "w") as f_b:
        for row in W:
            f_w.write(",".join(map(str, row)) + "\n")
        f_b.write(",".join(map(str, b)) + "\n")


save_weights_biases(W1, b1, "model_weights/W1.csv", "model_weights/b1.csv")
save_weights_biases(W2, b2, "model_weights/W2.csv", "model_weights/b2.csv")

# Testing the model
A1_test, A2_test = forward(X_test_scaled, W1, b1, W2, b2)
predictions = [1 if a > 0.5 else 0 for a in [row[0] for row in A2_test]]
accuracy = sum([1 for i in range(len(y_test)) if predictions[i] == y_test[i]]) / len(
    y_test
)
print(f"Test Accuracy: {accuracy}")
