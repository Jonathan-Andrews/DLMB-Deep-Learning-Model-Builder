import numpy as np
from layers.base_layer import Base_layer

import initializers as I
import regularizers as R 

class Batchnorm(Base_layer):
	def __init__(self, gamma_initializer="random", beta_initializer="random", gamma_regularizer="L0", beta_regularizer="L0", epsilon=1e-8,
		         trainable=True, *args, **kwargs):
		"""
		The Batchnorm class takes some data and scales and shifts it for better flow between the other layers,
		which solves problems like exploding or vanishing gradients.
		
		Arguments:
			gamma_initializer - str/instance of a function: An initializer function to set the value for the gamma var.
			beta_initializer - str/instance of a function: An initializer function to set the value for the beta var.
			gamma_regularizer -  str/instance of a regularizer class: A regularizer function that regularizes the gamma var.
			beta_regularizer -  str/instance of a regularizer class: A regularizer function that regularizes the beta var.
			epsilon - float/int: A very small number used in the calculations for the map_data() function.

		"""

		self.name = "Batchnorm"
		self.built = False
		self.trainable = trainable

		self.initializers = {"gamma":gamma_initializer, "beta":beta_initializer}
		self.regularizers = {"gamma":R.get(gamma_regularizer), "beta":R.get(beta_regularizer)}

		self.vars = {"gamma":0, "beta":0, "epsilon":epsilon}
		self.optimizations = {"gamma":[0,0], "beta":[0,0]}
		self.cached_data = {}

 	# ------------------------------------------------------------------------------------------------------------------------
	def set_vars(self):
		"""
		Sets the values for all of the vars that will be used.

		"""

		if not self.built:
			# Set self.vars["gamma"] based on self.initializers["gamma"].
			self.vars["gamma"] = I.get(self.initializers["gamma"])((1,1))

			# Set self.vars["beta"] based on self.initializers["beta"].
			self.vars["beta"] = I.get(self.initializers["beta"])((1,1))

		self.built = True

	# ------------------------------------------------------------------------------------------------------------------------
	def get_summary(self, with_vars, *args):
		"""
		get_summary() returns a summary of the layers features.

		Arguments:
			with_vars - bool: If True, get_summary() includes the layer's variables' values in the summary.

		Returns:
			summary - dict: A dictonary of the layers features.

		"""

		summary = {
					"name":self.name,
					"built":self.built,
					"trainable":self.trainable,
					"initializers":self.initializers,
					"regularizers":{key:self.regularizers[key].name for key in self.regularizers},
					"num_trainable_vars":2
				  }

		if with_vars:
			summary["vars"] = {key:np.asarray(self.vars[key]).tolist() for key in self.vars}
			summary["optimizations"] = {key:np.asarray(self.optimizations[key]).tolist() for key in self.optimizations}

		return summary

	# ------------------------------------------------------------------------------------------------------------------------
	def load(self, layer_data, *args, **kwargs):
		"""
		Takes the layer_data from the model this layer belongs to and sets all vars equal to each key in layer_data.

		Arguments:
			layer_data - dict: A dictonary of saved vars from when this layer was first built and then saved.

		"""

		self.name = layer_data["name"]
		self.built = layer_data["built"]
		self.trainable = layer_data["trainable"]
		self.initializers = layer_data["initializers"]
		self.regularizers = {key:R.get(layer_data["regularizers"][key]) for key in layer_data["regularizers"]}
		self.vars = {key:np.asarray(layer_data["vars"][key]) for key in layer_data["vars"]}
		self.optimizations = {key:np.asarray(layer_data["optimizations"][key]) for key in layer_data["optimizations"]}

 	# ------------------------------------------------------------------------------------------------------------------------
	def map_data(self, data):
		"""
		Maps the data to an output by shifting and scaling the incoming input.

		Arguments:
			data - Numpy array: A numpy array containing real numbers passed from the previous layer.

		Returns:
			output - Numpy array: A numpy array of outputs from this layer.

		"""

		self.cached_data["data"] = data

		self.cached_data["mu"] = np.mean(data, axis=1, keepdims=True)
		self.cached_data["var"] = np.var(data, axis=1, keepdims=True)
		self.cached_data["x_norm"] = (data-self.cached_data["mu"]) / np.sqrt(self.cached_data["var"]+self.vars["epsilon"])

		return self.vars["gamma"]*self.cached_data["x_norm"] + self.vars["beta"]

 	# ------------------------------------------------------------------------------------------------------------------------
	def calculate_gradients(self, error):
		"""
		Calculates the partial derivative of the error W.R.T each trainable var in the layer. 

		Arguments:
			error - Numpy array: A numpy array containing the error for the previous layer.

		Returns:
			output - Numpy array: A numpy array containing the error for this layer.

		"""

		# Calculate the beta gradients.
		self.optimizations["beta"][0] = np.mean(error) + self.regularizers["beta"].map_data(self.vars["beta"]) 

		# Calculate the gamma gradients.
		self.optimizations["gamma"][0] = np.mean(error*self.cached_data["x_norm"]) + self.regularizers["gamma"].map_data(self.vars["gamma"]) 

		# Calculate the partial derivative of the error W.R.T the output from the previous layer.
		dx_norm = error*self.vars["gamma"]
		var_inv = 1 / np.sqrt(self.cached_data["var"]+self.vars["epsilon"])
		x_mu = self.cached_data["data"]-self.cached_data["mu"]

		dvar = np.sum(dx_norm*x_mu, axis=1, keepdims=True) * -0.5 * (self.cached_data["var"]+self.vars["epsilon"])**(-3/2)
		dmu = np.sum(dx_norm*-var_inv, axis=1, keepdims=True) + dvar*1 / self.cached_data["data"].shape[1] * np.sum(-2*x_mu, axis=1, keepdims=True)

		# Return this layers error for the next layer to use.
		return (dx_norm*var_inv) + (dmu/self.cached_data["data"].shape[1]) + (dvar*2 / self.cached_data["data"].shape[1]*x_mu)

 	# ------------------------------------------------------------------------------------------------------------------------
	def update_vars(self, optimizer, epoch):
		"""
		Updates the trainable vars in the layer with the correct gradients.

		Arguments:
			optimizer - instance of a class: A class that takes each layer's gradients and optimizes them to reach a local optima in the error faster.
			epoch - int: The current epoch that the layer is training on.

		"""

		if self.trainable:
			self.optimizations["gamma"] = optimizer.map_data(self.optimizations["gamma"], epoch)
			self.optimizations["beta"] = optimizer.map_data(self.optimizations["beta"], epoch)

			self.vars["gamma"] -= self.optimizations["gamma"][0]
			self.vars["beta"] -= self.optimizations["beta"][0]
