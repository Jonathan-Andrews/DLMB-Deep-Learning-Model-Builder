from abc import ABCMeta, abstractmethod

class Base_layer(metaclass=ABCMeta):
	@abstractmethod
	def __init__(self):
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
	def map_data(self, data):
		"""
		Makes the layer map the incoming data to an output.

		Arguments:
			data - type = Numpy array: A matrix or vector (depending on batch size) containing real numbers passed from the previous layer.

		Returns:
			output - type = Numpy array: A matrix or vector (depending on batch size) of outputs from this layer.

		"""

		return output

 	# ------------------------------------------------------------------------------------------------------------------------
	@abstractmethod
	def calculate_gradients(self, error):
		"""
		Makes the layer calculate the gradients of the error W.R.T each trainable var in the layer.

		Arguments:
			error - type = Numpy array: A matrix or vector (depending on batch size) containing the error for the previous layer.

		Returns:
			output - type = Numpy array: A matrix or vector (depending on batch size) containing the error for this layer.

		"""

		return layer_error

 	# ------------------------------------------------------------------------------------------------------------------------
	@abstractmethod
	def update_vars(self, optimizer, epoch):
		"""
		Makes the layer update all the trainable vars with the previously calculated gradients.

		Arguments:
			optimizer - type = Instance of a class: A class that takes each layer's gradients and optimizes them to reach a local optima in the error faster.
			epoch - type = Int: The current epoch that the layer is training on.

		"""

		pass