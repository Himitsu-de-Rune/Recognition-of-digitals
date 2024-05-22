import numpy as np

def load_dataset():
	with np.load("mnist.npz") as f:
		# convert from RGB to Unit RGB
		x_train = f['x_train'].astype("float32") / 255

		# reshape from (60000, 28, 28) into (60000, 784)
		x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))

		# labels
		y_train = f['y_train']

		# convert to output layer format
		y_train = np.eye(10)[y_train]

		
		x_test = f['x_test'].astype("float32") / 255

		# reshape from (60000, 28, 28) into (60000, 784)
		x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))


		y_test = f['y_test']

		# convert to output layer format
		y_test = np.eye(10)[y_test]

		return x_train, y_train, x_test, y_test