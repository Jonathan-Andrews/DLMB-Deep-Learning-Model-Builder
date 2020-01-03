import os

import tensorflow as tf
import numpy as np

from models import Sequential
from layers import Dense, Batchnorm

def prepare_data():
	# Get the data from tensorflow
	# mnist is a data set of 60000 different 28*28 hand written digit images
	# The hand written digits range from 0 - 9
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

	# Set the shapes
	# The data set originally comes in a (60000, 28, 28) sized array, where each 28 is either the rows or columns of the image
	x_train = x_train.reshape((60000, 784))
	x_test = x_test.reshape((10000, 784))

	# Turn y_train into a one hot vector for softmax and crossentropy
	# A one hot vector is a array filled with zeros except for a one in the index of the number in the array
	# Example:
	# [2, 4]
	#   0  1  2  3  4    0  1  2  3  4
	# [[0, 0, 1, 0, 0], [0, 0, 0, 0, 1]] (The first 0 in each array is the zeroth index)
	y_train = np.eye(np.max(y_train) + 1)[y_train]

	# Normalize X.
	# We normalize the data to help the model learn better
	# A pixel in the image has a value ranging from 0 - 255, so dividing by 255 normalizes the data between 0 and 1
	x_train = x_train / 255
	x_test = x_test / 255

	return x_train, y_train, x_test, y_test





def validate_model(predictions, labels):
	correct = 0
	for i in range(len(predictions)):
		guess = np.where(predictions[i] == np.max(predictions[i]))
		if guess[0] == labels[i]: # Compare the models output to the correct output
			correct += 1

	return correct





def run():
	file_path = os.path.dirname(os.path.realpath(__file__)) + "/dlmb_mnist_example.json"

	# If a file of the neural-net model's architexture already exists,
	# then there is no need to build a new model.
	if os.path.isfile(file_path):

		# load the model and get its predictions based on x_test
		nn_model = Sequential()
		nn_model.load(file_path)

		predictions = nn_model.predict(x_test)

		# compare the predictions to the correct labels
		print(f"This model got a {validate_model(predictions, y_test)/100}% accuracy")


	# If the file doesn't exist then we need to build a neural-net model and train it.
	else:

		# Build the neural-net model
		nn_model = Sequential([
								Dense(128, 784, activation="ReLU"), # for the layer_dim we want 128 outputs and 784 inputs (each pixel on the image)
								Batchnorm(128),

								Dense(128, 128, activation="ReLU"),
								Batchnorm(128),

								Dense(32, 128, activation="ReLU"),
								Batchnorm(32),
		
								Dense(10, 32, activation="Softmax") # We have 10 nodes in the layer for each number from 0 - 9
							 ])

		nn_model.build(loss="crossentropy", optimizer="adam")
		# Crossentropy is a good loss function when you are doing logistic regression (classification)
		# Adam is one of the most popular optimizers

		nn_model.train(x_train, y_train, epochs=10, batch_size=1000)
		# Train the model
		# We go through the data 10 times and split the data of 60000 samples into 1000 sized batches leaving 60 samples

		# Now we save the model so we can use it again without re-training
		nn_model.save(file_path) # When saving, files must end in .json


x_train, y_train, x_test, y_test = prepare_data()

