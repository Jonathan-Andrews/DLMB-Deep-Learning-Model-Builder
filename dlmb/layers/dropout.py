import numpy as np
from layers.base_layer import Base_layer

class Dropout(Base_layer):
	def __init__(self, keep_prob=0.5):
		"""
		The Dropout class simplifies a neural-net model by randomly nulling some of the outputs from the previous layer.

		Arguments:
			keep_prob - type = float: A number between 0 and 1 that is a percentage of how many outputs to keep.
		
		"""

		self.layer_type = "Dropout"
		self.keep_prob = keep_prob

 	# ------------------------------------------------------------------------------------------------------------------------
	def set_vars(self):
		"""
		set_vars() is a function from the base_layer. This function can be passed because there are no vars to set up.

		"""

		pass

 	# ------------------------------------------------------------------------------------------------------------------------
	def map_data(self, data):
		"""
		map_data() makes a mask and applies it to the incoming input, randomly nulling some of the outputs.

		Arguments:
			data - type = Numpy array: A matrix or vector (depending on batch size) containing real numbers passed from the previous layer.

		Returns:
			output - type = Numpy array: A matrix or vector (depending on batch size) of outputs from this layer.

		"""

		self.mask = np.random.binomial(1, self.keep_prob, size=data.shape)/self.keep_prob
		return data*self.mask
    
    # ------------------------------------------------------------------------------------------------------------------------
	def calculate_gradients(self, error):
		"""
		Calculates the error for this layer woth the error from the previous layer.

		Arguments:
			error - type = Numpy array: A matrix or vector (depending on batch size) containing the error for the previous layer.

		Returns:
			output - type = Numpy array: A matrix or vector (depending on batch size) containing the error for this layer.

		"""

		return error*self.mask

 	# ------------------------------------------------------------------------------------------------------------------------
	def update_vars(self, optimizer, epoch):
		"""
		update_vars() is a function from the base_layer. This function can be passed because there are no vars to update.
	
		Arguments:
			optimizer - type = Instance of a class: A class that takes each layer's gradients and optimizes them to reach a local optima in the error faster.
			epoch - type = Int: The current epoch that the layer is training on.

		"""

		pass