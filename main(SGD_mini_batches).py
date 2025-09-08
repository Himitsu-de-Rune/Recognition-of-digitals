import numpy as np
import matplotlib.pyplot as plt

import utils

def train_val_test_split(X, y, val_size=0.2, test_size=0.1):
    total_size = X.shape[0]
    val_split = int(total_size * (1 - val_size - test_size))
    test_split = int(total_size * (1 - test_size))
    indices = np.arange(total_size)
    np.random.shuffle(indices)
    train_indices, val_indices, test_indices = indices[:val_split], indices[val_split:test_split], indices[test_split:]
    return X[train_indices], y[train_indices], X[val_indices], y[val_indices], X[test_indices], y[test_indices]

def create_mini_batches(X, y, batch_size):
    mini_batches = []
    data = np.hstack((X, y))
    np.random.shuffle(data)
    n_minibatches = data.shape[0] // batch_size
    for i in range(n_minibatches):
        mini_batch = data[i * batch_size:(i + 1) * batch_size, :]
        X_mini = mini_batch[:, :-10]  # Assuming the labels are one-hot encoded with 10 classes
        y_mini = mini_batch[:, -10:]
        mini_batches.append((X_mini, y_mini))
    if data.shape[0] % batch_size != 0:
        mini_batch = data[n_minibatches * batch_size:data.shape[0]]
        X_mini = mini_batch[:, :-10]
        y_mini = mini_batch[:, -10:]
        mini_batches.append((X_mini, y_mini))
    return mini_batches

images, labels, test_images, test_labels = utils.load_dataset()
train_images, train_labels, val_images, val_labels, test_images, test_labels = train_val_test_split(images, labels)

weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (20, 784))
weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (10, 20))
bias_input_to_hidden = np.zeros((20, 1))
bias_hidden_to_output = np.zeros((10, 1))

epochs = 5
learning_rate = 0.01
batch_size = 32

train_loss_vector = []
train_correct_vector = []
val_loss_vector = []
val_correct_vector = []

for epoch in range(epochs):
    print(f"Epoch №{epoch}")

    train_loss = 0
    train_correct = 0

    mini_batches = create_mini_batches(train_images, train_labels, batch_size)

    for batch_images, batch_labels in mini_batches:
        batch_delta_weights_input_to_hidden = np.zeros_like(weights_input_to_hidden)
        batch_delta_weights_hidden_to_output = np.zeros_like(weights_hidden_to_output)
        batch_delta_bias_input_to_hidden = np.zeros_like(bias_input_to_hidden)
        batch_delta_bias_hidden_to_output = np.zeros_like(bias_hidden_to_output)

        for image, label in zip(batch_images, batch_labels):
            image = np.reshape(image, (-1, 1))
            label = np.reshape(label, (-1, 1))

            hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
            hidden = 1 / (1 + np.exp(-hidden_raw))

            output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
            output = 1 / (1 + np.exp(-output_raw))

            train_loss += 1 / len(output) * np.sum((output - label) ** 2, axis=0)
            train_correct += int(np.argmax(output) == np.argmax(label))

            delta_output = output - label
            batch_delta_weights_hidden_to_output += delta_output @ np.transpose(hidden)
            batch_delta_bias_hidden_to_output += delta_output

            delta_hidden = np.transpose(weights_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
            batch_delta_weights_input_to_hidden += delta_hidden @ np.transpose(image)
            batch_delta_bias_input_to_hidden += delta_hidden

        weights_hidden_to_output += -learning_rate * batch_delta_weights_hidden_to_output / batch_size
        bias_hidden_to_output += -learning_rate * batch_delta_bias_hidden_to_output / batch_size
        weights_input_to_hidden += -learning_rate * batch_delta_weights_input_to_hidden / batch_size
        bias_input_to_hidden += -learning_rate * batch_delta_bias_input_to_hidden / batch_size

    train_loss_vector.append((train_loss[0] / train_images.shape[0]) * 100)
    train_correct_vector.append((train_correct / train_images.shape[0]) * 100)

    # Validate the model
    val_loss = 0
    val_correct = 0
    for image, label in zip(val_images, val_labels):
        image = np.reshape(image, (-1, 1))
        label = np.reshape(label, (-1, 1))

        hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
        hidden = 1 / (1 + np.exp(-hidden_raw)) 
        output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
        output = 1 / (1 + np.exp(-output_raw))

        val_loss += 1 / len(output) * np.sum((output - label) ** 2, axis=0)
        val_correct += int(np.argmax(output) == np.argmax(label))

    val_loss_vector.append((val_loss[0] / val_images.shape[0]) * 100)
    val_correct_vector.append((val_correct / val_images.shape[0]) * 100)

    print(f"Training Loss: {round((train_loss[0] / train_images.shape[0]) * 100, 3)}%")
    print(f"Training Accuracy: {round((train_correct / train_images.shape[0]) * 100, 3)}%")
    print(f"Validation Loss: {round((val_loss[0] / val_images.shape[0]) * 100, 3)}%")
    print(f"Validation Accuracy: {round((val_correct / val_images.shape[0]) * 100, 3)}%")
    print()

# Test the model
test_correct = 0
test_loss = 0

for image, label in zip(test_images, test_labels):
    image = np.reshape(image, (-1, 1))
    label = np.reshape(label, (-1, 1))

    hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
    hidden = 1 / (1 + np.exp(-hidden_raw))
    output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
    output = 1 / (1 + np.exp(-output_raw))

    test_loss += 1 / len(output) * np.sum((output - label) ** 2, axis=0)
    test_correct += int(np.argmax(output) == np.argmax(label))

print(f"Test Loss: {round((test_loss[0] / test_images.shape[0]) * 100, 3)}%")
print(f"Test Accuracy: {round((test_correct / test_images.shape[0]) * 100, 3)}%")

# Plot training and validation loss
fig, ax = plt.subplots()
x = np.linspace(0, epochs - 1, epochs)
ax.plot(x, train_loss_vector, label="Training Loss", linewidth=1.0)
ax.plot(x, val_loss_vector, label="Validation Loss", linewidth=1.0)
plt.title("Loss")
plt.legend()
plt.show()

# Plot training and validation accuracy
fig, ax = plt.subplots()
x = np.linspace(0, epochs - 1, epochs)
ax.plot(x, train_correct_vector, label="Training Accuracy", linewidth=1.0)
ax.plot(x, val_correct_vector, label="Validation Accuracy", linewidth=1.0)
plt.title("Accuracy")
plt.legend()
plt.show()

