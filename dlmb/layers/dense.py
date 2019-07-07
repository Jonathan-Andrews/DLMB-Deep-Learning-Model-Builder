import numpy as np
from layers.base_layer import Base_layer

import activations as A
import initializers as I
import regularizers as R 
# Always breath AIR. breathing AIR is good.

class Dense(Base_layer):
	def __init__(self, layer_shape, activation=A.Linear(), bias_type="per_node",
				 weight_initializer="uniform", bias_initializer="uniform",
				 weight_regularizer="L0", bias_regularizer="L0"):

		"""
		The dense class is one of the most simple neural-net layers. 
		It takes in an input and returns an output with the form of g(X*W + B).

		Arguments:
			layer_shape - type = tuple: A tuple of the layer shape, (E.X (6, 2), 6 is the number of nodes in the layer and 2 is the number of incoming inputs).
			activation - type = Instance of a class: An activation function that generally applies a non-linear function to the layer's output.
			bias_type - type = Str: Has three settings, per_node (a normal bias type), single (one bias weight for all nodes in this layer), none (no bias is used).
			weight_initializer - type = Str: An initializer function to set the values for the weights.
			bias_initializer - type = Str: An initializer function to set the values for the bias'.
			weight_regularizer - type = Str: A regularizer function that regularizes the weights.
			bias_regularizer - type = Str: A regularizer function that regularizes the bias'.

		"""

		self.layer_type = "Dense"
		self.built = False
		self.layer_shape = layer_shape
		self.activation = activation
		self.bias_type = bias_type
		self.weight_initializer = weight_initializer
		self.bias_initializer = bias_initializer
		self.weight_regularizer = weight_regularizer
		self.bias_regularizer = bias_regularizer

		# Stores the optimizations for the trainable vars.
		self.weight_optimizations = [0,0]
		self.bias_optimizations = [0,0]

		# Stores the weights and bias
		self.weights = []
		self.bias = []

		# Stores the mapped data in a dict.
		self.cached_vars = {}

 	# ------------------------------------------------------------------------------------------------------------------------
	def set_vars(self):
		"""
		Sets the values for all of the vars that will be used.
		Also sets the layers activation function if the user sent through a string as the argument.

		"""

		if self.built != True:
			# Sets the weights based on self.weight_initializer.
			if self.weight_initializer.lower() in ("zeros", "zero"):
				self.weights = I.zeros(self.layer_shape)
			elif self.weight_initializer.lower() in ("ones", "one"):
				self.weights = I.ones(self.layer_shape)
			elif self.weight_initializer.lower() in ("random"):
				self.weights = I.random(self.layer_shape)
			elif self.weight_initializer.lower() in ("uniform"):
				self.weights = I.uniform(self.layer_shape)
			elif self.weight_initializer.lower() in ("xavier"):
				self.weights = I.xavier(self.layer_shape)
			elif self.weight_initializer.lower() in ("xavier_uniform"):
				self.weights = I.xavier_uniform(self.layer_shape)
			elif self.weight_initializer.lower() in ("sigmoid_uniform"):
				self.weights = I.sigmoid_uniform(self.layer_shape)
			elif self.weight_initializer.lower() in ("relu"):
				self.weights = I.relu(self.layer_shape)
			elif self.weight_initializer.lower() in ("relu_uniform"):
				self.weights = I.relu_uniform(self.layer_shape)
			else:
				raise ValueError(f"{self.weight_initializer} is not currently an available weight initializer.")

			# Sets the bias' based on self.bias_initializer.
			if self.bias_initializer.lower() in ("zeros", "zero"):
				self.bias = I.zeros((1,1))
			elif self.bias_initializer.lower() in ("ones", "one"):
				self.bias = I.ones((1,1))
			elif self.bias_initializer.lower() in ("random"):
				self.bias = I.random((1,1))
			elif self.bias_initializer.lower() in ("uniform"):
				self.bias = I.uniform((1,1))
			elif self.bias_initializer.lower() in ("xavier"):
				self.bias = I.xavier((1,1))
			elif self.bias_initializer.lower() in ("xavier_uniform"):
				self.bias = I.xavier_uniform((1,1))
			elif self.bias_initializer.lower() in ("sigmoid_uniform"):
				self.bias = I.sigmoid_uniform((1,1))
			elif self.bias_initializer.lower() in ("relu"):
				self.bias = I.relu((1,1))
			elif self.bias_initializer.lower() in ("relu_uniform"):
				self.bias = I.relu_uniform((1,1))
			else:
				raise ValueError(f"{self.bias_initializer} is not currently an available bias initializer.")

		# Sets the layers activation function if the user sent through a string as the argument.
		if type(self.activation) == str:
			if self.activation.lower() in ("linear"):
				self.activation = A.Linear()
			elif self.activation.lower() in ("softmax"):
				self.activation = A.Softmax()
			elif self.activation.lower() in ("sigmoid"):
				self.activation = A.Sigmoid()
			elif self.activation.lower() in ("tanh"):
				self.activation = A.Tanh()
			elif self.activation.lower() in ("relu"):
				self.activation = A.ReLU()
			elif self.activation.lower() in ("leaky_relu"):
				self.activation = A.Leaky_ReLU()
			elif self.activation.lower() in ("elu"):
				self.activation = A.ELU()
			else:
				raise ValueError(f"{self.activation} is not currently an available activation function.")

		# Sets the regularizers if the user sent through a string as the argument.
		if type(self.weight_regularizer) == str:
			if self.weight_regularizer.lower() in ("l0"):
				self.weight_regularizer = R.L0()
			elif self.weight_regularizer.lower() in ("l1"):
				self.weight_regularizer = R.L1()
			elif self.weight_regularizer.lower() in ("l2"):
				self.weight_regularizer = R.L2()

		if type(self.bias_regularizer) == str:
			if self.bias_regularizer.lower() in ("l0"):
				self.bias_regularizer = R.L0()
			elif self.bias_regularizer.lower() in ("l1"):
				self.bias_regularizer = R.L1()
			elif self.bias_regularizer.lower() in ("l2"):
				self.bias_regularizer = R.L2()

		self.built = True

 	# ------------------------------------------------------------------------------------------------------------------------
	def map_data(self, data):
		"""
		Maps the data to an output in the form of g(X*W + B) where g(x) is self.activation.

		Arguments:
			data - type = Numpy array: A matrix or vector (depending on batch size) containing real numbers passed from the previous layer.

		Returns:
			output - type = Numpy array: A matrix or vector (depending on batch size) of outputs from this layer.

		"""

		# Makes sure that the data is the right shape.
		if len(data.shape) == 1:
			data = data.reshape((1, data.shape[0]))
		elif len(data.shape) == 2:
			data = data
		elif len(data.shape) == 3:
			data = data.reshape((data.shape[0], data.shape[1]*data.shape[2]))
		else:
			raise ValueError(f"data with size {data.shape} can not be used. Please reshape the data to a 1dim, 2dim or 3dim sized numpy array.")

		# Makes sure the size of the columns of the data is the same as the number of incoming inputs specified in self.layer_shape.
		if data.shape[1] == self.layer_shape[1]:
			self.cached_vars["data"] = data
			self.cached_vars["mapped_data"] = np.dot(data, self.weights.T) + self.bias.T
			return self.activation.function(self.cached_vars["mapped_data"])
		else:
			raise ValueError(f"The data with size {data.shape} can not be used. The columns must match the incoming inputs for the layer specified in self.layer_shape.")

 	# ------------------------------------------------------------------------------------------------------------------------
	def calculate_gradients(self, error):
		"""
		Calculates the partial derivative of the error W.R.T each trainable var in the layer. 
		In the dense layer the partial derivative of the error W.R.T each trainable var is dE/dO * dO/dZ * dZ/dW

		Arguments:
			error - type = Numpy array: A matrix or vector (depending on batch size) containing the error for the previous layer.

		Returns:
			output - type = Numpy array: A matrix or vector (depending on batch size) containing the error for this layer.

		"""

		dO_dZ = self.activation.derivative(self.cached_vars["mapped_data"])

		# Checks if the activation function was softmax, becuase the derivative of softmax returns a 3dim matrix.
		if len(dO_dZ.shape) == 3:
			error = error.reshape((error.shape[0], error.shape[1], 1))
			node_errors = np.sum(dO_dZ*error, axis=1)
		else:
			node_errors = dO_dZ*error

		# Calculate the weight gradients.
		self.weight_optimizations[0] = np.dot(node_errors.T, self.cached_vars["data"])/node_errors.shape[0] + self.weight_regularizer.regularize(self.weights)

		# Calculate the bias gradients based on self.bias_type.
		if self.bias_type == "per_node":
			self.bias_optimizations[0] = np.mean(node_errors.T, axis=1, keepdims=True) + self.bias_regularizer.regularize(self.bias)
		elif self.bias_type == "single":
			self.bias_optimizations[0] = np.mean(node_errors) + self.bias_regularizer.regularize(self.bias)
		elif self.bias_type == "none":
			self.bias_optimizations[0] = 0

		# Return this layer's error for the next layer to use.
		return np.dot(node_errors, self.weights)

 	# ------------------------------------------------------------------------------------------------------------------------
	def update_vars(self, optimizer, epoch):
		"""
		Updates the trainable vars in the layer with the correct gradients.

		Arguments:
			optimizer - type = Instance of a class: A class that takes each layer's gradients and optimizes them to reach a local optima in the error faster.
			epoch - type = Int: The current epoch that the layer is training on.

		"""
		self.weight_optimizations = optimizer.optimize(self.weight_optimizations, epoch)
		self.bias_optimizations = optimizer.optimize(self.bias_optimizations, epoch)

		self.weights = self.weights - self.weight_optimizations[0]
		self.bias = self.bias - self.bias_optimizations[0]