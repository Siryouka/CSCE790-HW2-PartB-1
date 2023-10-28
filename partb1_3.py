import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Initialize the neural network parameters
input_size = 2
output_size = 2
#hidden_size = 0  # We're using a one-layer network
learning_rate = 0.1
epochs = 101

# Define the input data
X = np.array([[0.1, 0.7, 0.8, 0.8, 1.0, 0.3, 0.0, -0.3, -0.5, -1.5],
              [1.2, 1.8, 1.6, 0.6, 0.8, 0.5, 0.2, 0.8, -1.5, -1.3]])

# Define the target data
Y = np.array([[1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])

# Initialize weights and biases
np.random.seed(0)
W = np.random.rand(output_size, input_size)
b = np.zeros((output_size, 1))


# Lists to store training error for plotting
errors = []
output = []
# Training loop
for epoch in range(epochs):
    total_error = 0

    for i in range(X.shape[1]):
        # Forward propagation
        Z = np.dot(W, X[:, i].reshape(-1, 1)) + b
        A = sigmoid(Z)
        output.append(A)

        # Calculate the error
        error = Y[:, i].reshape(-1, 1) - A
        total_error += np.sum(error ** 2)

        # Backpropagation
        dZ = error * A * (1 - A)
        dW = np.dot(dZ, X[:, i].reshape(1, -1))
        db = dZ

        # Update weights and biases
        W += learning_rate * dW
        b += learning_rate * db

    errors.append(total_error/X.shape[1])

    color_labels = ["r", "r", "r", "b", "b", "y", "y", "y", "g", "g"]
    if epoch in [3, 10, 100]:
        # Plot the decision boundary and data points
        plt.figure()
        plt.title(f"Epoch {epoch}")
        for cc in range(len(color_labels)):
            plt.scatter(X[0, cc], X[1, cc], c=color_labels[cc])
        plt.xlabel("Input 1")
        plt.ylabel("Input 2")

        # x1 = np.linspace(-2, 2, 100)
        # x2 = -(W[0, 0] * x1 + b[0]) / W[0, 1]
        # x3 = -(W[0, 1] * x1 + b[1]) / W[1, 0]
        # x4 = -(W[1, 0] * x1 + b[0]) / W[1, 1]
        # x5 = -(W[1, 1] * x1 + b[1]) / W[0, 0]
        # plt.plot(x1, x2, 'r--')
        # plt.plot(x1, x3, 'r--')
        # plt.plot(x1, x4, 'r--')
        # plt.plot(x1, x5, 'r--')

        x1 = np.linspace(-2, 2, 100)
        x21 = -(W[0, 0] * x1 + b[0]) / W[0, 1]
        x22 = -(W[1, 0] * x1 + b[1]) / W[1, 1]

        plt.plot(x1, x21, 'r--')
        plt.plot(x1, x22, 'r--')


plt.figure()
plt.plot(range(epochs), errors)
plt.title("Training Error vs. Epoch Number")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.show()
