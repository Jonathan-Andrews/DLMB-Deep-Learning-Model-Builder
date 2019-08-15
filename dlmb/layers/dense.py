import numpy as np
from layers.base_layer import Base_layer

import activations as A
import initializers as I
import regularizers as R 

class Dense(Base_layer):
	def __init__(self, output_size, input_shape=1, bias_type="per_node", trainable=True, activation="Linear",
				 weight_initializer="uniform", bias_initializer="uniform",
	             weight_regularizer="L0", bias_regularizer="L0", *args, **kwargs):

		"""
		The Dense layer is one of the most simplest layers. 

		Arguments:
			output_size - int: The number of outputs for this layer.
			input_shape - int/tuple: The shape (excluding batch size) of the input data this layer will receive.
			bias_type - string: Has three settings, per_node (a bias with weights for every output), single (one bias weight for all outputs), none (no bias is used).
			trainable - bool: If set to True, the vars in this layer will update based on the calculated loss of this layer W.R.T the vars.
			activation - instance of an activation class: A mathematical function that generally applies a non-linear mapping to the functions input.
			weight_initializer - str/instance of a function: An initializer function to set the values for the weights.
			bias_initializer - str/instance of a function: An initializer function to set the values for the bias'.
			weight_regularizer - str/instance of a regularizer class: A regularizer function that regularizes the weights.
			bias_regularizer - str/instance of a regularizer class: A regularizer function that regularizes the bias'.

		"""

		# Flattens the input_shape if data type is tuple.
		if type(input_shape) == tuple:
			shape = 1
			for i in range(len(input_shape)):
				shape *= input_shape[i]
			input_shape = shape

		self.name = "Dense"
		self.built = False
		self.layer_shape = (output_size, input_shape)
		self.bias_type = bias_type
		self.trainable = trainable

		self.activation = A.get(activation)
		self.initializers = {"weight":weight_initializer, "bias":bias_initializer}
		self.regularizers = {"weight":R.get(weight_regularizer), "bias":R.get(bias_regularizer)}

		self.vars = {"weight":0, "bias":0}
		self.optimizations = {"weight":[0,0], "bias":[0,0]}
		self.cached_data = {}

 	# ------------------------------------------------------------------------------------------------------------------------
	def set_vars(self, *args, **kwargs):
		"""
		Sets the values for all of the vars that will be used.
		Also sets the layers activation function if the user sent through a string as the argument.

		"""

		if not self.built:
			# Set self.vars["weight"] based on self.initializers["weight"].
			self.vars["weight"] = I.get(self.initializers["weight"])(self.layer_shape)

			# Set self.vars["bias"] based on self.initializers["bias"].
			self.vars["bias"] = I.get(self.initializers["bias"])((self.layer_shape[0], 1))

			# Set self.vars["bias"] based on self.bias_type.
			if self.bias_type == "per_node":
				self.vars["bias"] = self.vars["bias"]
			elif self.bias_type == "single":
				self.vars["bias"] = self.vars["bias"][0]
			elif self.bias_type == "none":
				self.vars["bias"] == np.array([[0]])
			else:
				print("At Dense.set_vars(): '%s' is not a valid bias_type. Has been set to 'per_node' by default" % self.bias_type)
		else:
			for key in self.vars:
				self.vars[key] = np.asarray(self.vars[key])

		self.built = True

	# ------------------------------------------------------------------------------------------------------------------------
	def get_summary(self, with_vars, *args, **kwargs):
		"""
		get_summary() returns a summary of the layers features.

		Arguments:
			with_vars - type = Bool: If True, get_summary() includes the layer's variables' values in the summary.

		Returns:
			summary - type = Dict: A dictonary of the layers features.

		"""

		summary = {
					"name":self.name,
					"built":self.built,
					"layer_shape":self.layer_shape,
					"bias_type":self.bias_type,
					"trainable":self.trainable,
					"activation":self.activation.name,
					"initializers":self.initializers,
					"regularizers":{key:self.regularizers[key].name for key in self.regularizers},
					"num_trainable_vars":(self.vars["weight"].shape[0]*self.vars["weight"].shape[1])+self.vars["bias"].shape[0]
				  }

		if with_vars:
			summary["vars"] = {key:np.asarray(self.vars[key]).tolist() for key in self.vars}
			summary["optimizations"] = {key:np.asarray(self.optimizations[key]).tolist() for key in self.optimizations}

		return summary

	# ------------------------------------------------------------------------------------------------------------------------
	def load(self, layer_data, *args, **kwargs):
		"""
		Takes the layer_data from the model this layer belongs to, and sets all vars equal to each key in layer_data.

		Arguments:
			layer_data - dict: A dictonary of saved vars from when this layer was first built and then saved.

		"""

		self.name = layer_data["name"]
		self.built = layer_data["built"]
		self.layer_shape = layer_data["layer_shape"]
		self.bias_type = layer_data["bias_type"]
		self.trainable = layer_data["trainable"]
		self.activation = A.get(layer_data["activation"])
		self.initializers = layer_data["initializers"]
		self.regularizers = {key:R.get(layer_data["regularizers"][key]) for key in layer_data["regularizers"]}
		self.vars = {key:np.asarray(layer_data["vars"][key]) for key in layer_data["vars"]}
		self.optimizations = {key:np.asarray(layer_data["optimizations"][key]) for key in layer_data["optimizations"]}

 	# ------------------------------------------------------------------------------------------------------------------------
	def map_data(self, data, *args, **kwargs):
		"""
		Makes the layer map the incoming data to an output.

		Arguments:
			data - Numpy array: A numpy array containing real numbers passed from the previous layer.

		Returns:
			output - Numpy array: A numpy array of outputs from this layer.

		"""

		# Makes sure that the data is the right shape.
		if len(data.shape) == 1:
			data = data.reshape((1, data.shape[0]))
		elif len(data.shape) > 1:
			length = 1
			for i in range(len(data.shape)-1):
				length *= data.shape[i+1]
			data = data.reshape((data.shape[0], length))

		if self.layer_shape[1] == data.shape[1]:
			self.cached_data["data"] = data
			self.cached_data["mapped_data"] = np.dot(data, self.vars["weight"].T) + self.vars["bias"].T
			return self.activation.map_data(self.cached_data["mapped_data"])
		else:
			raise ValueError("At Dense.map_data(): Data with shape %s does not match this layers specified input_shape of %s" % (data.shape, self.layer_shape[1]))

 	# ------------------------------------------------------------------------------------------------------------------------
	def calculate_gradients(self, error, *args, **kwargs):
		"""
		Calculates the gradients of the error W.R.T each trainable var in the layer.
		In the dense layer the partial derivative of the error W.R.T each trainable var is dE/dO * dO/dZ * dZ/dW

		Arguments:
			error - Numpy array: A numpy array containing the error for the previous layer.

		Returns:
			output - Numpy array: A numpy array containing the error for this layer.

		"""

		dO_dZ = self.activation.calculate_gradients(self.cached_data["mapped_data"])

		# Checks if the activation function was softmax, becuase the derivative of softmax returns a 3dim numpy array.
		if len(dO_dZ.shape) == 3:
			error = error.reshape((error.shape[0], error.shape[1], 1))
			node_errors = np.sum(dO_dZ*error, axis=1)
		else:
			node_errors = dO_dZ*error

		# Calculate the weight gradients.
		self.optimizations["weight"][0] = np.dot(node_errors.T, self.cached_data["data"])/node_errors.shape[0] + self.regularizers["weight"].map_data(self.vars["weight"])

		# Calculate the bias gradients based on self.bias_type.
		if self.bias_type == "per_node":
			self.optimizations["bias"][0] = np.mean(node_errors.T, axis=1, keepdims=True) + self.regularizers["bias"].map_data(self.vars["bias"])
		elif self.bias_type == "single":
			self.optimizations["bias"][0] = np.mean(np.sum(node_errors, axis=1)) + self.regularizers["bias"].map_data(self.vars["bias"])
		else:
			self.optimizations["bias"][0] = np.array([[0.0]])

		# Return this layer's error for the next layer to use.
		return np.dot(node_errors, self.vars["weight"])

 	# ------------------------------------------------------------------------------------------------------------------------
	def update_vars(self, optimizer, epoch, *args, **kwargs):
		"""
		Updates the trainable vars in the layer with the correct gradients.

		Arguments:
			optimizer - instance of a class: A class that takes each layer's gradients and optimizes them to reach a local optima in the error faster.
			epoch - int: The current epoch that the layer is training on.

		"""

		if self.trainable:
			self.optimizations["weight"] = optimizer.map_data(self.optimizations["weight"], epoch)
			self.optimizations["bias"] = optimizer.map_data(self.optimizations["bias"], epoch)

			self.vars["weight"] -= self.optimizations["weight"][0]
			self.vars["bias"] -= self.optimizations["bias"][0]
