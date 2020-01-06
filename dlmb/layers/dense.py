from layers.base_layer import *
from utils.function_helpers import *

import activations as a
import initializers as i


class Dense(Base_Layer):
	@accepts(self="any", output_size=int, input_shape=(tuple, int), activation=(a.Base_Activation, str),
			 bias_type=str, trainable=bool, weight_initializer=str, bias_initializer=str)

	def __init__(self, output_size, input_shape, activation='linear', bias_type="per_node", 
				 trainable=True, weight_initializer="uniform", bias_initializer="uniform") -> None:

		"""
			The Dense layer is one of the most simplest layers. 

			Arguments:
				output_size        : int                 : The number of outputs for this layer.
				input_shape        : tuple/int      	 : The shape (excluding batch size) of the input data this layer will receive.
				activation         : Base_Activation/str : A mathematical function that generally applies a non-linear mapping to some data.
				bias_type          : str      	         : Has three settings, per_node (a bias with weights for every output), 
												         - single (one bias weight for all outputs), none (no bias is used).
				trainable          : bool     	         : If True, the vars in this layer will update based on the calculated loss of this layer W.R.T the vars.
				weight_initializer : str                 : An initializer function to set the values for the weights.
				bias_initializer   : st                  : An initializer function to set the values for the bias'.

		"""

		super().__init__('Dense')

		# Flattens the input_shape if data type is tuple.
		if type(input_shape) == tuple:
			input_shape = len(np.zeros(input_shape).flatten())

		self.built = False
		self.layer_shape = (output_size, input_shape)
		self.bias_type = bias_type
		self.trainable = trainable

		self.activation = a.get(activation)

		self.weights = 0
		self.bias = 0

		self.weight_initializer = weight_initializer
		self.bias_initializer = bias_initializer

		self.optimizations = {"weight":[0,0], "bias":[0,0]}
		self.cached_data = {}

 	
	def build(self) -> None:
		"""
			Sets the values for all of the vars that will be used.
			Also sets the layers activation function if the user sent through a string as the argument.

		"""

		if not self.built:
			
			# Set self.weights.
			self.weights = i.get(self.weight_initializer)(self.layer_shape)

			# Set self.bias.
			self.bias = i.get(self.weight_initializer)((self.layer_shape[0], 1))

			# Set self.bias based on self.bias_type.
			if self.bias_type == "per_node":
				self.bias = self.bias
			elif self.bias_type == "single":
				self.bias = self.bias[0]
			elif self.bias_type == "none":
				self.bias == np.array([[0]])
			else:
				print("At Dense.set_vars(): '%s' is not a valid bias_type. Has been set to 'per_node' by default" % self.bias_type)

		self.built = True

	
	@accepts(self="any", with_vars=bool)
	def get_summary(self, with_vars) -> dict:
		"""
			get_summary() returns a summary of the layers features.

			Arguments:
				with_vars : bool : If True, get_summary() includes the layer's variables' values in the summary.

			Returns:
				summary : dict : A dictonary of the layers features.

		"""

		summary = {
					"name":self.name,
					"built":self.built,
					"layer_shape":self.layer_shape,
					"bias_type":self.bias_type,
					"trainable":self.trainable,
					"activation":self.activation.name,
					"weight_initializer":self.weight_initializer,
					"bias_initializer":self.bias_initializer,
					"num_trainable_vars":(self.weights.shape[0]*self.weights.shape[1])+self.bias.shape[0]
				  }

		if with_vars:
			summary["weights"] =  np.asarray(self.weights).tolist()
			summary["bias"] =  np.asarray(self.bias).tolist()
			summary["optimizations"] = {key:np.asarray(self.optimizations[key]).tolist() for key in self.optimizations}

		return summary

	
	@accepts(self="any", layer_data=dict)
	def load(self, layer_data) -> None:
		"""
			Takes the layer_data from the model this layer belongs to, and sets all vars equal to each key in layer_data.

			Arguments:
				layer_data : dict : A dictonary of saved vars from when this layer was first built and then saved.

		"""

		self.name = layer_data["name"]
		self.built = layer_data["built"]
		self.layer_shape = tuple(layer_data["layer_shape"])
		self.bias_type = layer_data["bias_type"]
		self.trainable = layer_data["trainable"]
		self.activation = a.get(layer_data["activation"])
		self.weight_initializer = layer_data["weight_initializer"]
		self.bias_initializer = layer_data["bias_initializer"]
		self.weights = np.asarray(layer_data["weights"])
		self.bias = np.asarray(layer_data["bias"])
		self.optimizations = {key:np.asarray(layer_data["optimizations"][key]) for key in layer_data["optimizations"]}
 

	@accepts(self="any", data=np.ndarray)
	def map_data(self, data) -> np.ndarray:
		"""
			Maps the incoming input to an output in the form of f(x*w + b).

			Arguments:
				data : np.ndarray : An n dimensional numpy array containing real numbers passed from the previous layer.

			Returns:
				output : np.ndarray : An n dimensional numpy array of outputs from this layer.

		"""

		# Try to write a decorator for this.
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
			self.cached_data["mapped_data"] = np.dot(data, self.weights.T) + self.bias.T
			return self.activation.map_data(self.cached_data["mapped_data"])
		else:
			raise ValueError("At Dense.map_data(): Data with shape %s does not match this layers specified input_shape of %s" % (data.shape[1], self.layer_shape[1]))


	@accepts(self="any", error=np.ndarray)
	def calculate_gradients(self, error) -> np.ndarray:
		"""
			Calculates the gradients of the error W.R.T each trainable var in the layer.
			In the dense layer the partial derivative of the error W.R.T each trainable var is,
			dError/dOutput * dOutput/dInput * dInput/dVar.

			Arguments:
				error : np.ndarray : An n dimensional numpy array containing the errors for the previous layer.

			Returns:
				output : np.ndarray : An n dimensional numpy array containing the errors for this layer.

		"""

		# error = dError_dOutput.
		# Calculate dOuput_dInput.
		dO_dI = self.activation.calculate_gradients(self.cached_data["mapped_data"])

		# Checks if the activation function was softmax, becuase the derivative of softmax returns a 3d numpy array.
		# We want a 2d numpy array.
		if self.activation.name == "softmax":
			error = error.reshape((error.shape[0], error.shape[1], 1))
			node_errors = np.sum(error*dO_dI, axis=1)
		else:
			node_errors = error*dO_dI

		# Calculate the weight gradients, dInput/dVar.
		self.optimizations["weight"][0] = np.dot(node_errors.T, self.cached_data["data"])/node_errors.shape[0] # Take the average

		# Calculate the bias gradients based on self.bias_type.
		if self.bias_type == "per_node":
			self.optimizations["bias"][0] = np.mean(node_errors.T, axis=1, keepdims=True)
		elif self.bias_type == "single":
			self.optimizations["bias"][0] = np.mean(np.sum(node_errors, axis=1))
		else:
			self.optimizations["bias"][0] = np.array([[0.0]])

		# Return this layer's error for the next layer to use.
		return np.dot(node_errors, self.weights)

 	
	@accepts(self="any", optimizer=op.Base_Optimizer, epoch=int)
	def update_vars(self, optimizer, epoch) -> None:
		"""
			Updates the trainable vars in the layer with the correct gradients.

			Arguments:
				optimizer : base_optimizer : An optimizer class that takes each layer's gradients and optimizes them to reach a local optima in the error faster.
				epoch     : int       	   : The current epoch that the layer is training on.

		"""

		if self.trainable:
			self.optimizations["weight"] = optimizer.map_data(self.optimizations["weight"], epoch)
			self.optimizations["bias"] = optimizer.map_data(self.optimizations["bias"], epoch)

			self.weights -= self.optimizations["weight"][0]
			self.bias -= self.optimizations["bias"][0]
