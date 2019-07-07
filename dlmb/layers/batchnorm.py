import numpy as np
from layers.base_layer import Base_layer

import initializers as I
import regularizers as R 

class Batchnorm(Base_layer):
	def __init__(self, beta_initializer="random", gamma_initializer="random", beta_regularizer="L0", gamma_regularizer="L0", epsilon=1e-8):
		"""
		The Batchnorm class takes some data and scales and shifts it for better flow between the other layers.
		The Batchnorm class solves problems like exploding or vanishing gradients.
		
		Arguments:
			beta_initializer - type = Str: An initializer function to set the value for the beta var.
			gamma_initializer - type = Str: An initializer function to set the value for the gamma var.
			beta_regularizer - type = Str: A regularizer function that regularizes the beta var.
			gamma_regularizer - type = Str: A regularizer function that regularizes the gamma var.
			epsilon - type = Float: A very small number used in the calculations for the map_data() function.

		"""

		self.layer_type = "Batchnorm"
		self.built = False
		self.beta_initializer = beta_initializer
		self.gamma_initializer = gamma_initializer
		self.beta_regularizer = beta_regularizer
		self.gamma_regularizer = gamma_regularizer
		self.epsilon = epsilon

		# Stores the optimizations for the trainable vars.
		self.beta_optimizations = [0, 0]
		self.gamma_optimizations = [0, 0]

		# Stores the beta and the gamma
		self.beta = []
		self.gamma = []

		# Stores the mapped data in a dict.
		self.cached_vars = {}

 	# ------------------------------------------------------------------------------------------------------------------------
	def set_vars(self):
		"""
		Sets the values for all of the vars that will be used.

		"""

		if self.built != True:
			# Sets the value for the beta var based on self.beta_initializer.
			if self.beta_initializer.lower() in ("zeros", "zero"):
				self.beta = I.zeros((1,1))
			elif self.beta_initializer.lower() in ("ones", "one"):
				self.beta = I.ones((1,1))
			elif self.beta_initializer.lower() in ("random"):
				self.beta = I.random((1,1))
			elif self.beta_initializer.lower() in ("uniform"):
				self.beta = I.uniform((1,1))
			elif self.beta_initializer.lower() in ("xavier"):
				self.beta = I.xavier((1,1))
			elif self.beta_initializer.lower() in ("xavier_uniform"):
				self.beta = I.xavier_uniform((1,1))
			elif self.beta_initializer.lower() in ("sigmoid_uniform"):
				self.beta = I.sigmoid_uniform((1,1))
			elif self.beta_initializer.lower() in ("relu"):
				self.beta = I.relu((1,1))
			elif self.beta_initializer.lower() in ("relu_uniform"):
				self.beta = I.relu_uniform((1,1))
			else:
				raise ValueError(f"{self.beta_initializer} is not an available beta initializer")

			# Sets the value for the gamma var based on self.gamma_initializer.
			if self.gamma_initializer.lower() in ("zeros", "zero"):
				self.gamma = I.zeros((1,1))
			elif self.gamma_initializer.lower() in ("ones", "one"):
				self.gamma = I.ones((1,1))
			elif self.gamma_initializer.lower() in ("random"):
				self.gamma = I.random((1,1))
			elif self.gamma_initializer.lower() in ("uniform"):
				self.gamma = I.uniform((1,1))
			elif self.gamma_initializer.lower() in ("xavier"):
				self.gamma = I.xavier((1,1))
			elif self.gamma_initializer.lower() in ("xavier_uniform"):
				self.gamma = I.xavier_uniform((1,1))
			elif self.gamma_initializer.lower() in ("sigmoid_uniform"):
				self.gamma = I.sigmoid_uniform((1,1))
			elif self.gamma_initializer.lower() in ("relu"):
				self.gamma = I.relu((1,1))
			elif self.gamma_initializer.lower() in ("relu_uniform"):
				self.gamma = I.relu_uniform((1,1))
			else:
				raise ValueError(f"{self.gamma_initializer} is not an available gamma initializer")

		# Sets the regularizers if the user sent through a string as the argument.
		if type(self.beta_regularizer) == str:
			if self.beta_regularizer.lower() in ("l0"):
				self.beta_regularizer = R.L0()
			elif self.beta_regularizer.lower() in ("l1"):
				self.beta_regularizer = R.L1()
			elif self.beta_regularizer.lower() in ("l2"):
				self.beta_regularizer = R.L2()

		if type(self.gamma_regularizer) == str:
			if self.gamma_regularizer.lower() in ("l0"):
				self.gamma_regularizer = R.L0()
			elif self.gamma_regularizer.lower() in ("l1"):
				self.gamma_regularizer = R.L1()
			elif self.gamma_regularizer.lower() in ("l2"):
				self.gamma_regularizer = R.L2()

		self.built = True

 	# ------------------------------------------------------------------------------------------------------------------------
	def map_data(self, data):
		"""
		Maps the data to an output by shifting and scaling the incoming input.

		Arguments:
			data - type = Numpy array: A matrix or vector (depending on batch size) containing real numbers passed from the previous layer.

		Returns:
			output - type = Numpy array: A matrix or vector (depending on batch size) of outputs from this layer.

		"""

		self.cached_vars["data"] = data

		self.cached_vars["mu"] = np.mean(data, axis=1, keepdims=True)
		self.cached_vars["var"] = np.var(data, axis=1, keepdims=True)
		self.cached_vars["x_norm"] = (data-self.cached_vars["mu"]) / np.sqrt(self.cached_vars["var"]+self.epsilon)

		return self.gamma*self.cached_vars["x_norm"] + self.beta

 	# ------------------------------------------------------------------------------------------------------------------------
	def calculate_gradients(self, error):
		"""
		Calculates the partial derivative of the error W.R.T each trainable var in the layer. 

		Arguments:
			error - type = Numpy array: A matrix or vector (depending on batch size) containing the error for the previous layer.

		Returns:
			output - type = Numpy array: A matrix or vector (depending on batch size) containing the error for this layer.

		"""

		# Calculate the beta gradients.
		self.beta_optimizations[0] = np.mean(error) # + self.beta_regularizer.function(self.beta) 

		# Calculate the gamma gradients.
		self.gamma_optimizations[0] = np.mean(error*self.cached_vars["x_norm"]) # + self.gamma_regularizer.function(self.gamma) 

		# Calculate the partial derivative of the error W.R.T the output from the previous layer.
		dx_norm = error*self.gamma
		var_inv = 1 / np.sqrt(self.cached_vars["var"]+self.epsilon)
		x_mu = self.cached_vars["data"]-self.cached_vars["mu"]

		dvar = np.sum(dx_norm*x_mu, axis=1, keepdims=True) * -0.5 * (self.cached_vars["var"]+self.epsilon)**(-3/2)
		dmu = np.sum(dx_norm*-var_inv, axis=1, keepdims=True) + dvar*1 / self.cached_vars["data"].shape[1] * np.sum(-2*x_mu, axis=1, keepdims=True)

		# Return this layers error for the next layer to use.
		return (dx_norm*var_inv) + (dmu/self.cached_vars["data"].shape[1]) + (dvar*2 / self.cached_vars["data"].shape[1]*x_mu)

 	# ------------------------------------------------------------------------------------------------------------------------
	def update_vars(self, optimizer, epoch):
		"""
		Updates the trainable vars in the layer with the correct gradients.

		Arguments:
			optimizer - type = Instance of a class: A class that takes each layer's gradients and optimizes them to reach a local optima in the error faster.
			epoch - type = Int: The current epoch that the layer is training on.

		"""

		self.beta_optimizations = optimizer.optimize(self.beta_optimizations, epoch)
		self.gamma_optimizations = optimizer.optimize(self.gamma_optimizations, epoch)

		self.beta -= self.beta_optimizations[0]
		self.gamma -= self.gamma_optimizations[0]