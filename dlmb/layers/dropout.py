import numpy as np
from layers.base_layer import Base_layer

class Dropout(Base_layer):
	def __init__(self, keep_prob=0.5):
		"""
		The Dropout class simplifies a neural-net model by randomly nulling some of the outputs from the previous layer.

		Arguments:
			keep_prob - float: A number between 0 and 1 that is a percentage of how many outputs to keep.
		
		"""

		self.name = "Dropout"
		self.keep_prob = keep_prob

 	# ------------------------------------------------------------------------------------------------------------------------
	def set_vars(self):
		"""
		set_vars() is a function from the base_layer. This function can be passed because there are no vars to set up.

		"""

		pass

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
					"keep_prob":self.keep_prob,
					"num_trainable_vars":0
				  }

		return summary

	# ------------------------------------------------------------------------------------------------------------------------
	def load(self, layer_data, *args, **kwargs):
		"""
		Takes the layer_data from the model this layer belongs to and sets all vars equal to each key in layer_data.

		Arguments:
			layer_data - dict: A dictonary of saved vars from when this layer was first built and then saved.

		"""

		self.name = layer_data["name"]
		self.keep_prob = layer_data["keep_prob"]

 	# ------------------------------------------------------------------------------------------------------------------------
	def map_data(self, data):
		"""
		map_data() makes a mask and applies it to the incoming input, randomly nulling some of the outputs.

		Arguments:
			data - Numpy array: A numpy array containing real numbers passed from the previous layer.

		Returns:
			output - Numpy array: A numpy array of outputs from this layer.
			
		"""

		self.mask = np.random.binomial(1, self.keep_prob, size=data.shape)/self.keep_prob
		return data*self.mask
    
    # ------------------------------------------------------------------------------------------------------------------------
	def calculate_gradients(self, error):
		"""
		Calculates the error for this layer W.R.T the error from the previous layer.

		Arguments:
			error - Numpy array: A numpy array containing the error for the previous layer.

		Returns:
			output - Numpy array: A numpy array containing the error for this layer.

		"""

		return error*self.mask

 	# ------------------------------------------------------------------------------------------------------------------------
	def update_vars(self, optimizer, epoch):
		"""
		update_vars() is a function from the base_layer. This function can be passed because there are no vars to update.
	
		Arguments:
			optimizer - instance of a class: A class that takes each layer's gradients and optimizes them to reach a local optima in the error faster.
			epoch - int: The current epoch that the layer is training on.

		"""

		pass
