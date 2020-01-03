from abc import ABCMeta, abstractmethod
import numpy as np

import optimizers as op


class Base_Layer(metaclass=ABCMeta):
	@abstractmethod
	def __init__(self, name:str, **kwargs) -> None:
		"""
			The Base_Layer class is an abstract class for all neural-net layers. 
			All neural-net layers must inherit from Base_layer.

		"""

		self.name = name


	@abstractmethod
	def build(self) -> None:
		"""
			Sets all of the vars that will be used in the layer.

		"""

		pass

	
	@abstractmethod
	def get_summary(self, with_vars:bool) -> dict:
		"""
			get_summary() returns a summary of the layers features.

			Arguments:
				with_vars : bool : If True, get_summary() includes the layer's variables' values in the summary.

			Returns:
				summary : dict : A dictonary of the layers features.

		"""

		return summary

	
	@abstractmethod
	def load(self, layer_data:dict) -> None:
		"""
			Takes the layer_data from the model this layer belongs to, and sets all vars equal to each key in layer_data.

			Arguments:
				layer_data : dict : A dictonary of saved vars from when this layer was first built and then saved.

		"""

		pass


	@abstractmethod
	def map_data(self, data:np.ndarray) -> np.ndarray:
		"""
			Maps the data from the previous layer to an output.

			Arguments:
				data : np.ndarray : An n dimensional numpy array containing real numbers passed from the previous layer.

			Returns:
				output : np.ndarray : An n dimensional numpy array of outputs from this layer.

		"""

		return output


	@abstractmethod
	def calculate_gradients(self, error:np.ndarray) -> np.ndarray:
		"""
			Calculates the derivatives of the error from the previous layer W.R.T each trainable var in this layer.

			Arguments:
				error : np.ndarray : An n dimensional numpy array containing the errors for the previous layer.

			Returns:
				output : np.ndarray : An n dimensional numpy array containing the errors for this layer.

		"""

		return layer_error


	@abstractmethod
	def update_vars(self, optimizer:op.Base_Optimizer, epoch:int) -> None:
		"""
			Updates all the trainable vars with the previously calculated gradients.

			Arguments:
				optimizer : Base_Optimizer : The optimization function the user wants to use.
				epoch : int : The current epoch or time step the model is at.

		"""

		pass