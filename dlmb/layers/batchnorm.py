from layers.base_layer import *
from utils.function_helpers import *

import initializers as i


class Batchnorm(Base_Layer):
	@accepts(self="any", input_shape=(tuple, int), epsilon=float, trainable=bool,
			 gamma_initializer=str, beta_initializer=str)

	def __init__(self, input_shape, epsilon=1e-8, trainable=True, 
		         gamma_initializer="uniform", beta_initializer="uniform") -> None:
		"""
			The Batchnorm class takes some data and scales and shifts it for better flow between the other layers,
			which solves problems like exploding or vanishing gradients.
		
			Arguments:
				epsilon   : float : A very small number used in the calculations for the map_data() function.
				trainable : bool  : If True, the vars in this layer will update based on the calculated loss of this layer W.R.T the vars.

		"""

		super().__init__('Batchnorm')

		# Flattens the input_shape if data type is tuple.
		if type(input_shape) == tuple:
			input_shape = len(np.zeros(input_shape).flatten())


		self.layer_shape = (input_shape, input_shape)
		self.built = False
		self.trainable = trainable

		self.gamma = 0
		self.beta = 0
		self.epsilon = epsilon

		self.gamma_initializer = gamma_initializer
		self.beta_initializer = beta_initializer

		self.optimizations = {"gamma":[0,0], "beta":[0,0]}
		self.cached_data = {}


	def build(self) -> None:
		"""
			Sets the values for all of the vars that will be used.

		"""

		if not self.built:

			# Set self.gamma based on self.gamma_initializer.
			self.gamma = i.get(self.gamma_initializer)((1, self.layer_shape[0]))

			# Set self.beta based on self.beta_initializer.
			self.beta = i.get(self.beta_initializer)((1, self.layer_shape[0]))

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
					"trainable":self.trainable,
					"epsilon":self.epsilon,
					"gamma_initializer":self.gamma_initializer,
					"beta_initializer":self.beta_initializer,
					"num_trainable_vars":2
				  }

		if with_vars:
			summary["gamma"] = self.gamma.tolist()
			summary["beta"] = self.beta.tolist()
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
		self.trainable = layer_data["trainable"]
		self.epsilon = layer_data["epsilon"]
		self.gamma_initializer = layer_data["gamma_initializer"]
		self.beta_initializer = layer_data["beta_initializer"]
		self.gamma = np.asarray(layer_data["gamma"])
		self.beta = np.asarray(layer_data["beta"])
		self.optimizations = {key:np.asarray(layer_data["optimizations"][key]) for key in layer_data["optimizations"]}


	@accepts(self="any", data=np.ndarray)
	def map_data(self, data) -> np.ndarray:
		"""
			Maps the data by mathematically shifting and scalling it.

			Arguments:
				data : np.ndarray : An n dimensional numpy array containing real numbers passed from the previous layer.

			Returns:
				output : np.ndarray : An n dimensional numpy array of outputs from this layer.

		"""

		data_shape = data.shape

		# Makes sure that the data is the right shape.
		if len(data.shape) == 1:
			data = data.reshape((1, data.shape[0]))
		elif len(data.shape) > 1:
			length = 1
			for i in range(len(data.shape)-1):
				length *= data.shape[i+1]
			data = data.reshape((data.shape[0], length))


		self.cached_data["data"] = data

		self.cached_data["mean"] = np.mean(data, axis=0)
		self.cached_data["var"] = np.var(data, axis=0)

		self.cached_data["x_norm"] = division_check(data-self.cached_data["mean"], np.sqrt(self.cached_data["var"]+self.epsilon))
		return self.cached_data["x_norm"]*self.gamma + self.beta

	

	@accepts(self="any", error=np.ndarray)
	def calculate_gradients(self, error) -> np.ndarray:
		"""
			Calculates the gradients of the error W.R.T each trainable var in the layer.

			Arguments:
				error : np.ndarray : An n dimensional numpy array containing the errors for the previous layer.

			Returns:
				output : np.ndarray : An n dimensional numpy array containing the errors for this layer.

		"""

		error_shape = error.shape

		# Makes sure that the error is the right shape.
		if len(error.shape) == 1:
			error = error.reshape((1, error.shape[0]))
		elif len(error.shape) > 1:
			length = 1
			for i in range(len(error.shape)-1):
				length *= error.shape[i+1]
			error = error.reshape((error.shape[0], length))


		# Calculate the beta gradients.
		self.optimizations["beta"][0] = np.sum(error, axis=0, keepdims=True)

		# Calculate the gamma gradients.
		self.optimizations["gamma"][0] = np.sum(error*self.cached_data["x_norm"], axis=0, keepdims=True)

		# Calculate the derivative of the error w.r.t the input
		part1 = division_check(self.gamma, error.shape[0])
		part2 = np.sqrt(division_check(1, self.cached_data["var"]+self.epsilon))

		part3 = (-1*self.optimizations["gamma"][0])*self.cached_data["x_norm"]
		part4 = error.shape[0] * error
		part5 = np.dot(-1*np.ones((error.shape[0], 1)), self.optimizations["beta"][0])

		return part1 * part2 * (part3 + part4 + part5)



	@accepts(self="any", optimizer=op.Base_Optimizer, epoch=int)
	def update_vars(self, optimizer, epoch) -> None:
		"""
			Updates the trainable vars in the layer with the correct gradients.

			Arguments:
				optimizer : base_optimizer : An optimizer class that takes each layer's gradients and optimizes them to reach a local optima in the error faster.
				epoch     : int       	   : The current epoch that the layer is training on.

		"""

		if self.trainable:
			self.optimizations["gamma"] = optimizer.map_data(self.optimizations["gamma"], epoch)
			self.optimizations["beta"] = optimizer.map_data(self.optimizations["beta"], epoch)

			self.gamma = self.gamma - self.optimizations["gamma"][0]
			self.beta = self.beta - self.optimizations["beta"][0]
