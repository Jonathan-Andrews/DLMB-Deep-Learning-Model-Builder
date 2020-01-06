from abc import ABCMeta, abstractmethod
import numpy as np

from utils.function_helpers import *


class Base_Loss(metaclass=ABCMeta):
	@abstractmethod
	def __init__(self) -> None:
		"""
			The Base_Loss class is an abstract class for all loss functions. 
			All loss functions must inherit from Base_Loss.

		"""

		pass

	
	@abstractmethod
	def map_data(self, y_true:np.ndarray, y_pred:np.ndarray) -> np.ndarray:
		"""
			map_data() takes some data and applies a mathematical mapping to it.
		
			Arguments:
				y_true : np.ndarray : An n dimensional numpy array of target values for the output of a neural-net model.
				y_pred : np.ndarray : An n dimensional numpy array of predicted values from a neural-net model.


			Return:
				output : np.ndarray : An n dimensional numpy array of the mapped data.

		"""

		return output


	@abstractmethod
	def calculate_gradients(self, y_true:np.ndarray, y_pred:np.ndarray) -> np.ndarray:
		"""
			calculate_gradients returns the derivative of the loss function W.R.T the data.
	
			Arguments:
				y_true : np.ndarray : An n dimensional numpy array of target values for the output of a neural-net model.
				y_pred : np.ndarray : An n dimensional numpy array of predicted values from a neural-net model.


			Return:
				output : np.ndarray : An n dimensional numpy array of gradients.

		"""

		return output





class Mean_Squared_Error(Base_Loss):
	def __init__(self) -> None:
		"""
			The MSE class is a commonly used regression loss function. 

		"""

		pass


	@accepts(self="any", y_true=np.ndarray, y_pred=np.ndarray)
	def map_data(self, y_true, y_pred) -> np.ndarray:
		"""
		Calculates the squared distance between y_true and y_pred.

		Arguments:
			y_true : np.ndarray : An n dimensional numpy array of target values for the output of a neural-net model.
			y_pred : np.ndarray : An n dimensional numpy array of predicted values from a neural-net model.

		Returns:
			output : np.ndarray : An n dimensional numpy array of the mean squared distance between y_true and y_pred.

		"""

		return (y_pred-y_true)**2/2


	@accepts(self="any", y_true=np.ndarray, y_pred=np.ndarray)
	def calculate_gradients(self, y_true, y_pred) -> np.ndarray:
		"""
		Calculates the derivatives of the function W.R.T y_pred.

		Arguments:
			y_true : np.ndarray : An n dimensional numpy array of target values for the output of a neural-net model.
			y_pred : np.ndarray : An n dimensional numpy array of predicted values from a neural-net model.

		Returns:
			output : np.ndarray : An n dimensional numpy array of the calculated derivatives of the function W.R.T y_pred.

		"""

		return y_pred - y_true





class Binary_Crossentropy(Base_Loss):
	def __init__(self) -> None:
		"""
			The Binary_crossentropy class measures the performance of a classification model whose output is a probability value between 0 and 1, 
			and where the number of outputs is less than 3.

		"""

		pass


	@accepts(self="any", y_true=np.ndarray, y_pred=np.ndarray)
	def map_data(self, y_true, y_pred) -> np.ndarray:
		"""
			Calculates the distance between y_true and y_pred.
	
			Arguments:
				y_true : np.ndarray : An n dimensional numpy array of target values for the output of a neural-net model.
				y_pred : np.ndarray : An n dimensional numpy array of predicted values from a neural-net model.

			Returns:
				output : np.ndarray : The mean squared distance between y_true and y_pred.

		"""

		part1 = y_true*np.log(y_pred+1.0e-8) # I add 1.0e-8 to make sure 0 isn't going into np.log
		part2 = (1-y_true)*np.log(1-y_pred+1.0e-8)
		return -(part1 + part2)


	@accepts(self="any", y_true=np.ndarray, y_pred=np.ndarray)
	def calculate_gradients(self, y_true, y_pred) -> np.ndarray:
		"""
			Calculates the derivatives of the function W.R.T y_pred.

			Arguments:
				y_true : np.ndarray : An n dimensional numpy array of target values for the output of a neural-net model.
				y_pred : np.ndarray : An n dimensional numpy array of predicted values from a neural-net model.

			Returns:
				output : np.ndarray : An n dimensional numpy array of the calculated derivatives of the function W.R.T y_pred.

		"""

		return division_check(y_true,y_pred) - division_check(1-y_true, 1-y_pred)





class Crossentropy(Base_Loss):
	def __init__(self) -> None:
		"""
			The Crossentropy class measures the performance of a classification model whose output is a probability value between 0 and 1, 
			and where the number of outputs is more than 2.

		"""

		pass


	@accepts(self="any", y_true=np.ndarray, y_pred=np.ndarray)
	def map_data(self, y_true, y_pred) -> np.ndarray:
		"""
			Calculates the distance between y_true and y_pred.

			Arguments:
				y_true : np.ndarray : An n dimensional numpy array of target values for the output of a neural-net model.
				y_pred : np.ndarray : An n dimensional numpy array of predicted values from a neural-net model.

			Returns:
				output : np.ndarray : The mean squared distance between y_true and y_pred.

		"""

		return -(y_true*np.log(y_pred+1.0e-8))


	def calculate_gradients(self, y_pred:np.ndarray, y_true:np.ndarray) -> np.ndarray:
		"""
		Calculates the derivatives of the function W.R.T y_pred.

		Arguments:
			y_true : np.ndarray : An n dimensional numpy array of target values for the output of a neural-net model.
			y_pred : np.ndarray : An n dimensional numpy array of predicted values from a neural-net model.

		Returns:
			output : np.ndarray : An n dimensional numpy array of the calculated derivatives of the function W.R.T y_pred.

		"""

		return division_check(y_true, y_pred)





def get(loss) -> Base_Loss:
	"""
		Finds and returns the correct loss function.

		Arguments:
			loss : Base_Loss/str : The loss function the user wants to use.

		Returns:
			loss : Base_Loss : The correct loss function.
		
	"""

	if isinstance(loss, str):
		if loss.lower() in ("mse", "mean_squared_error"):
			return Mean_Squared_Error()
		elif loss.lower() in ("bc", "bce", "binary_crossentropy"):
			return Binary_Crossentropy()
		elif loss.lower() in ("ce", "crossentropy"):
			return Crossentropy()
		else:
			print("At losses.get(): '%s' is not an available loss function. Has been set to 'Mean_squared_error' by default" % loss)
			return Mean_squared_error()
	elif isinstance(loss, Base_Loss):
		return loss
	else:
		raise ValueError("At losses.get(): Expected 'class inheriting from Base_Loss' or 'str' for the argument 'loss', recieved '%s'" % type(loss))
