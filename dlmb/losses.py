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
	def map_data(self, data):
		"""
		This is where the math happens, the function takes some data and applies a mathematical mapping to it.
	
		Arguments:
			data - Numpy array: The data that the function will be mapping to an output.

		Return:
			output - Numpy array: The mapped data.

		"""

		return output

	# ------------------------------------------------------------------------------------------------------------------------
	@abstractmethod
	def calculate_gradients(self, data):
		"""
		Calculates the derivative of the loss function.
	
		Arguments:
			data - Numpy array: The data that the derivatives will be calculated W.R.T.

		Return:
			output - Numpy array: The calculated derivatives.

		"""

		return output

# ------------------------------------------------------------------------------------------------------------------------
class Mean_squared_error(Base_loss):
	def __init__(self):
		"""
		The MSE class is a commonly used regression loss function. MSE is the mean squared distances between the target values and predicted values.

		"""

		pass

	# ------------------------------------------------------------------------------------------------------------------------
	def map_data(self, y_pred, y_true):
		"""
		Calculates the squared distance between y_true and y_pred.

		Arguments:
			y_pred - Numpy array: A numpy array of predicted values from the model.
			y_true - Numpy array: A numpy array of target values for the output.

		Returns:
			output - Numpy array: The calculated mean squared distance between y_true and y_pred.

		"""

		return 2**(y_pred - y_true)/2

	# ------------------------------------------------------------------------------------------------------------------------
	def calculate_gradients(self, y_pred, y_true):
		"""
		Calculates the derivatives of the function W.R.T y_pred.

		Arguments:
			y_pred - Numpy array: A numpy array of predicted values from the model.
			y_true - Numpy array: A numpy array of target values for the output.

		Returns:
			output - Numpy array: The calculated derivatives of the function with respect to y_pred.

		"""

		return y_pred - y_true

# ------------------------------------------------------------------------------------------------------------------------
class Binary_crossentropy(Base_loss):
	def __init__(self):
		"""
		The Binary_crossentropy class measures the performance of a classification model whose output is a probability value between 0 and 1, 
		and where the number of outputs is less than 3.

		"""

		pass

	# ------------------------------------------------------------------------------------------------------------------------
	def map_data(self, y_pred, y_true):
		"""
		Calculates the distance between y_true and y_pred.

		Arguments:
			y_pred - Numpy array: A numpy array of predicted values from the model.
			y_true - Numpy array: A numpy array of target values for the output.

		Returns:
			output - Numpy array: The calculated distance between y_true and y_pred.

		"""

		return -1 * (y_true * np.log(y_pred+1.0e-8) + (1-y_true) * np.log(1-y_pred+1.0e-8))

	# ------------------------------------------------------------------------------------------------------------------------
	def calculate_gradients(self, y_pred, y_true):
		"""
		Calculates the derivatives of the function W.R.T y_pred.

		Arguments:
			y_pred - Numpy array: A numpy array of predicted values from the model.
			y_true - Numpy array: A numpy array of target values for the output.

		Returns:
			output - Numpy array: The calculated derivatives of the function with respect to y_pred.

		"""

		return -1 * ((y_true-y_pred) / (y_pred*(-y_pred+1))+1.0e-8)

# ------------------------------------------------------------------------------------------------------------------------
class Crossentropy(Base_loss):
	def __init__(self):
		"""
		The Crossentropy class measures the performance of a classification model whose output is a probability value between 0 and 1, 
		and where the number of outputs is more than 2.

		"""

		pass

	# ------------------------------------------------------------------------------------------------------------------------
	def map_data(self, y_pred, y_true):
		"""
		Calculates the distance between y_true and y_pred.

		Arguments:
			y_pred - Numpy array: A numpy array of predicted values from the model.
			y_true - Numpy array: A numpy array of target values for the output.

		Returns:
			output - Numpy array: The calculated distance between y_true and y_pred.

		"""

		return -1 * (y_true*np.log(y_pred+1.0e-8))

	# ------------------------------------------------------------------------------------------------------------------------
	def calculate_gradients(self, y_pred, y_true):
		"""
		Calculates the derivatives of the function W.R.T y_pred.

		Arguments:
			y_pred - Numpy array: A numpy array of predicted values from the model.
			y_true - Numpy array: A numpy array of target values for the output.

		Returns:
			output - Numpy array: The calculated derivatives of the function with respect to y_pred.

		"""

		return -1 * (y_true/(y_pred+1.0e-8))

# ------------------------------------------------------------------------------------------------------------------------
def get(loss):
	"""
	Finds and returns the correct loss class.

	Arguments:
		loss - str/instance of a class: The loss class.

	Returns:
		loss - instance of a class: The correct loss class.
		
	"""

	if type(loss) == str:
		if loss.lower() in ("mse", "mean_squared_error"):
			return Mean_squared_error()
		elif loss.lower() in ("bc", "binary_crossentropy"):
			return Binary_crossentropy()
		elif loss.lower() in ("ce", "crossentropy"):
			return Crossentropy()
		else:
			print("'%s' is not currently an available loss function. Has been set to 'Mean_squared_error' by default" % loss)
			return MSE()
	else:
		return loss
