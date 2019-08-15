from abc import ABCMeta, abstractmethod

class Base_layer(metaclass=ABCMeta):
	@abstractmethod
	def __init__(self, *args, **kwargs):
		"""
		The Base_layer class is an abstract class and makes sure every layer uses the functions down below.

		"""

		pass

 	# ------------------------------------------------------------------------------------------------------------------------
	@abstractmethod
	def set_vars(self):
		"""
		Makes the layer set all of the vars that will be used.

		"""

		pass

	# ------------------------------------------------------------------------------------------------------------------------
	@abstractmethod
	def get_summary(self, with_vars):
		"""
		get_summary() returns a summary of the layers features.

		Arguments:
			with_vars - bool: If True, get_summary() includes the layer's variables' values in the summary.

		Returns:
			summary - dict: A dictonary of the layers features.

		"""

		return summary

	# ------------------------------------------------------------------------------------------------------------------------
	@abstractmethod
	def load(self, layer_data):
		"""
		Takes the layer_data from the model this layer belongs to, and sets all vars equal to each key in layer_data.

		Arguments:
			layer_data - dict: A dictonary of saved vars from when this layer was first built and then saved.

		"""

		pass

 	# ------------------------------------------------------------------------------------------------------------------------
	@abstractmethod
	def map_data(self, data):
		"""
		Makes the layer map the incoming data to an output.

		Arguments:
			data - Numpy array: A numpy array containing real numbers passed from the previous layer.

		Returns:
			output - Numpy array: A numpy array of outputs from this layer.

		"""

		return output

 	# ------------------------------------------------------------------------------------------------------------------------
	@abstractmethod
	def calculate_gradients(self, error):
		"""
		Makes the layer calculate the gradients of the error W.R.T each trainable var in the layer.

		Arguments:
			error - Numpy array: A numpy array containing the error for the previous layer.

		Returns:
			output - Numpy array: A numpy array containing the error for this layer.

		"""

		return layer_error

 	# ------------------------------------------------------------------------------------------------------------------------
	@abstractmethod
	def update_vars(self, optimizer, epoch):
		"""
		Makes the layer update all the trainable vars with the previously calculated gradients.

		Arguments:
			optimizer - instance of a class: A class that takes each layer's gradients and optimizes them to reach a local optima in the error faster.
			epoch - int: The current epoch that the layer is training on.

		"""

		pass
