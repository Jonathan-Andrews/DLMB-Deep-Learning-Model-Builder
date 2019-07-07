from abc import ABCMeta, abstractmethod
import numpy as np

# ------------------------------------------------------------------------------------------------------------------------
class Base_loss(metaclass=ABCMeta):
	@abstractmethod
	def __init__(self):
		"""
		The Base_loss class is an abstract class and makes sure every loss function uses the functions down below.

		"""

		pass

	# ------------------------------------------------------------------------------------------------------------------------
	@abstractmethod
	def function(self, data):
		"""
		This is where the math happens. The function takes some data and applies a mathematical mapping to it.
	
		Arguments:
			data - type = Numpy array: The data that the function will be mapping to an output.

		Return:
			output - type = Numpy array: The mapped data.

		"""

		return output

	# ------------------------------------------------------------------------------------------------------------------------
	@abstractmethod
	def derivative(self, data):
		"""
		Calculates the derivative of the loss function.
	
		Arguments:
			data - type = Numpy array: The data that the derivative will be calculate with respect to.

		Return:
			output - type = Numpy array: The calculated derivative.

		"""

		return output

# ------------------------------------------------------------------------------------------------------------------------
class Mean_squared_error(Base_loss):
	def __init__(self):
		"""
		The MSE class is a commonly used regression loss function. MSE is the squared distances between the target values and predicted values.

		"""

		pass

	# ------------------------------------------------------------------------------------------------------------------------
	def function(self, y_pred, y_true):
		"""
		Calculates the squared distance between y_true and y_pred.

		Arguments:
			y_pred - type = Numpy array: A matrix or vector (depending on batch_size) of predicted values from the model.
			y_true - type = Numpy array: A matrix or vector (depending on batch size) of target values for the output.

		Returns:
			output - type = Numpy array: The calculated squared distance between y_true and y_pred.

		"""

		return 2**(y_pred - y_true)/2

	# ------------------------------------------------------------------------------------------------------------------------
	def derivative(self, y_pred, y_true):
		"""
		Calculates the derivative of the function with respect to y_pred.

		Arguments:
			y_pred - type = Numpy array: A matrix or vector (depending on batch_size) of predicted values from the model.
			y_true - type = Numpy array: A matrix or vector (depending on batch size) of target variables for the output.

		Returns:
			output - type = Numpy array: The calculated derivative of the function with respect to y_pred.

		"""

		return y_pred - y_true

# ------------------------------------------------------------------------------------------------------------------------
class Binary_crossentropy(Base_loss):
	def __init__(self):
		"""
		The Binary_crossentropy class measures the performance of a classification model whose output is a probability value between 0 and 1, 
		and where the number of data points is less than 3.

		"""

		pass

	# ------------------------------------------------------------------------------------------------------------------------
	def function(self, y_pred, y_true):
		"""
		Calculates the distance between y_true and y_pred.

		Arguments:
			y_pred - type = Numpy array: A matrix or vector (depending on batch_size) of predicted values from the model.
			y_true - type = Numpy array: A matrix or vector (depending on batch size) of target values for the output.

		Returns:
			output - type = Numpy array: The calculated distance between y_true and y_pred.

		"""

		return -1 * (y_true * np.log(y_pred+1.0e-8) + (1-y_true) * np.log(1-y_pred+1.0e-8))

	# ------------------------------------------------------------------------------------------------------------------------
	def derivative(self, y_pred, y_true):
		"""
		Calculates the derivative of the function with respect to y_pred.

		Arguments:
			y_pred - type = Numpy array: A matrix or vector (depending on batch_size) of predicted values from the model.
			y_true - type = Numpy array: A matrix or vector (depending on batch size) of target variables for the output.

		Returns:
			output - type = Numpy array: The calculated derivative of the function with respect to y_pred.

		"""

		return -1 * ((y_true-y_pred) / (y_pred*(-y_pred+1))+1.0e-8)

# ------------------------------------------------------------------------------------------------------------------------
class Crossentropy(Base_loss):
	def __init__(self):
		"""
		The Crossentropy class measures the performance of a classification model whose output is a probability value between 0 and 1, 
		and where the number of data points is more than 2.

		"""

		pass

	# ------------------------------------------------------------------------------------------------------------------------
	def function(self, y_pred, y_true):
		"""
		Calculates the distance between y_true and y_pred.

		Arguments:
			y_pred - type = Numpy array: A matrix or vector (depending on batch_size) of predicted values from the model.
			y_true - type = Numpy array: A matrix or vector (depending on batch size) of target values for the output.

		Returns:
			output - type = Numpy array: The calculated distance between y_true and y_pred.

		"""

		return -1 * (y_true*np.log(y_pred+1.0e-8))

	# ------------------------------------------------------------------------------------------------------------------------
	def derivative(self, y_pred, y_true):
		"""
		Calculates the derivative of the function with respect to y_pred.

		Arguments:
			y_pred - type = Numpy array: A matrix or vector (depending on batch_size) of predicted values from the model.
			y_true - type = Numpy array: A matrix or vector (depending on batch size) of target variables for the output.

		Returns:
			output - type = Numpy array: The calculated derivative of the function with respect to y_pred.

		"""

		return -1 * (y_true/(y_pred+1.0e-8))