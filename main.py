import numpy as np
import matplotlib.pyplot as plt

import utils

images, labels, test_images, test_labels = utils.load_dataset()

weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (20, 784))
weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (10, 20))
bias_input_to_hidden = np.zeros((20, 1))
bias_hidden_to_output = np.zeros((10, 1))

epochs = 10
e_loss = 0
e_loss_vector = []
e_correct = 0
e_correct_vector = []
learning_rate = 0.01

for epoch in range(epochs):
	print(f"Epoch №{epoch}")

	for image, label in zip(images, labels):
		image = np.reshape(image, (-1, 1))
		label = np.reshape(label, (-1, 1))

		# Forward propagation (to hidden layer)
		hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
		hidden = 1 / (1 + np.exp(-hidden_raw)) # sigmoid

		# Forward propagation (to output layer)
		output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
		output = 1 / (1 + np.exp(-output_raw))

		# Loss / Error calculation
		e_loss += 1 / len(output) * np.sum((output - label) ** 2, axis=0)
		e_correct += int(np.argmax(output) == np.argmax(label))

		# Backpropagation (output layer)
		delta_output = output - label
		weights_hidden_to_output += -learning_rate * delta_output @ np.transpose(hidden)
		bias_hidden_to_output += -learning_rate * delta_output

		# Backpropagation (hidden layer)
		delta_hidden = np.transpose(weights_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
		weights_input_to_hidden += -learning_rate * delta_hidden @ np.transpose(image)
		bias_input_to_hidden += -learning_rate * delta_hidden

		# DONE

	e_loss_vector.append(e_loss[0])
	e_correct_vector.append(e_correct)

	# print some debug info between epochs
	print(f"Loss: {round((e_loss[0] / images.shape[0]) * 100, 3)}%")
	print(f"Accuracy: {round((e_correct / images.shape[0]) * 100, 3)}%")
	print()
	e_loss = 0
	e_correct = 0


count = 0

for image, label in zip(test_images, test_labels):
	image = np.reshape(image, (-1, 1))
	label = np.reshape(label, (-1, 1))

	# Forward propagation (to hidden layer)
	hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
	hidden = 1 / (1 + np.exp(-hidden_raw)) # sigmoid
	# Forward propagation (to output layer)
	output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
	output = 1 / (1 + np.exp(-output_raw))

	if count <= 3:
		print(f"label: {np.argmax(label)}")
		plt.imshow(image.reshape(28, 28), cmap="Greys")
		plt.title(f"NN suggests the CUSTOM number is: {output.argmax()}")
		plt.show()

	count += 1
	e_correct += int(np.argmax(output) == np.argmax(label))

fig, ax = plt.subplots()
x = np.linspace(0, epochs-1, epochs)
ax.plot(x, e_loss_vector, linewidth=1.0)
plt.title("loss")
plt.show()

fig, ax = plt.subplots()
x = np.linspace(0, epochs-1, epochs)
ax.plot(x, e_correct_vector, linewidth=1.0)
plt.title("accuracy")
plt.show()

print()
print("Test:")
print(f"Accuracy: {round(e_correct / test_images.shape[0] * 100, 3)}%")
