from abc import ABCMeta, abstractmethod
import numpy as np

from utils.function_helpers import *


class Base_Activation(metaclass=ABCMeta):
	@abstractmethod
	def __init__(self) -> None:
		"""
			The Base_Activation class is an abstract class for all activation functions. 
			All activation functions must inherit from Base_Activation.

		"""

		self.name = "Base_Activation"



	@abstractmethod
	def map_data(self, data) -> np.ndarray:
		"""
			map_data() takes some data and applies a mathematical mapping to it.
	
			Arguments:
				data : np.ndarray : An n dimensional numpy array of data that the function will be mapping to an output.

			Return:
				output : np.ndarray : An n dimensional numpy array of the mapped data.

		"""

		return output

	
	
	@abstractmethod
	def calculate_gradients(self, data) -> np.ndarray:
		"""
			Calculates the derivative of the activation function.
		
			Arguments:
				data : np.ndarray : An n dimensional numpy array of data that the derivative will be calculated W.R.T.

			Return:
				output : np.ndarray : An n dimensional numpy array of the calculated derivative.

		"""

		return output





class Linear(Base_Activation):
	def __init__(self) -> None:
		"""
			The Linear class is the default activation function and generally isn't actually used.

		"""

		self.name = "linear"


	@accepts(self="any", data=np.ndarray)	
	def map_data(self, data) -> np.ndarray:
		"""
			Maps some data to an output with the form of f(x) = x.

			Arguments:
				data : np.ndarray : An n dimensional numpy array of data that the function will be mapping to an output.

			Return:
				output : np.ndarray : An n dimensional numpy array of the mapped data.

		"""

		return data

	
	@accepts(self="any", data=np.ndarray)
	def calculate_gradients(self, data) -> np.ndarray:
		"""
			Calculates the derivative of the activation function.
		
			Arguments:
				data : np.ndarray : An n dimensional numpy array of data that the derivative will be calculated W.R.T.

			Return:
				output : np.ndarray : An n dimensional numpy array of the calculated derivative.

		"""

		return np.ones_like(data)





class Softmax(Base_Activation):
	def __init__(self) -> None:
		"""
			The Softmax class takes an array and normalizes it into a probability distribution with the same size.
			Generally used for the output layer.

		"""

		self.name = "softmax"


	@accepts(self="any", data=np.ndarray)	
	def map_data(self, data) -> np.ndarray:
		"""
			Maps some data to an output with the form of f(x) = e^x_k / sum(e^x_i).

			Arguments:
				data : np.ndarray : An n dimensional numpy array of data that the function will be mapping to an output.

			Return:
				output : np.ndarray : An n dimensional numpy array of the mapped data.

		"""

		e_x = np.exp(data-np.max(data))
		return division_check(e_x, np.sum(e_x, axis=1, keepdims=True))

	
	@accepts(self="any", data=np.ndarray)
	def calculate_gradients(self, data) -> np.ndarray:
		"""
			Calculates the derivative of the activation function.
		
			Arguments:
				data : np.ndarray : An n dimensional numpy array of data that the derivative will be calculated W.R.T.

			Return:
				output : np.ndarray : An n dimensional numpy array of the calculated derivative.

		"""

		a = np.reshape(self.map_data(data), (data.shape[0], data.shape[1], 1))
		e = np.ones((a.shape[1], 1))
		i = np.identity(a.shape[1])
		return (a*e.T) * (i - e*a.reshape((a.shape[0], a.shape[2], a.shape[1])))





class Sigmoid(Base_Activation):
	def __init__(self) -> None:
		"""
			The Sigmoid class squashes some data between a range of 0 and 1. 
			Good for probabilities and generally used for any layer.

		"""

		self.name = "sigmoid"


	@accepts(self="any", data=np.ndarray)	
	def map_data(self, data) -> np.ndarray:
		"""
			Maps some data to an output with the form of f(x) = 1/(1+e^-x).

			Arguments:
				data : np.ndarray : An n dimensional numpy array of data that the function will be mapping to an output.

			Return:
				output : np.ndarray : An n dimensional numpy array of the mapped data.

		"""

		return division_check(1, 1+np.exp(-data))

	
	@accepts(self="any", data=np.ndarray)
	def calculate_gradients(self, data) -> np.ndarray:
		"""
			Calculates the derivative of the activation function.
		
			Arguments:
				data : np.ndarray : An n dimensional numpy array of data that the derivative will be calculated W.R.T.

			Return:
				output : np.ndarray : An n dimensional numpy array of the calculated derivative.

		"""

		return self.map_data(data) * (1-self.map_data(data))





class Tanh(Base_Activation):
	def __init__(self) -> None:
		"""
			The Tanh class is the hyperbolic tangent function. 
			Squashes some data between a range of -1 and 1.

		"""

		self.name = "tanh"


	@accepts(self="any", data=np.ndarray)	
	def map_data(self, data) -> np.ndarray:
		"""
			Maps some data to an output with the form of f(x) = sinh(x)/cosh(x).

			Arguments:
				data : np.ndarray : An n dimensional numpy array of data that the function will be mapping to an output.

			Return:
				output : np.ndarray : An n dimensional numpy array of the mapped data.

		"""

		return np.tanh(data) # Numpy already has a tahn function.

	
	@accepts(self="any", data=np.ndarray)
	def calculate_gradients(self, data) -> np.ndarray:
		"""
			Calculates the derivative of the activation function.
		
			Arguments:
				data : np.ndarray : An n dimensional numpy array of data that the derivative will be calculated W.R.T.

			Return:
				output : np.ndarray : An n dimensional numpy array of the calculated derivative.

		"""

		return 1-self.map_data(data)**2





class ReLU(Base_Activation):
	def __init__(self) -> None:
		"""
			The ReLU class is the Rectified Linear Unit function. 
			Commonly used for hidden layers.

		"""

		self.name = "relu"


	@accepts(self="any", data=np.ndarray)	
	def map_data(self, data) -> np.ndarray:
		"""
			Maps some data to an output with the form of f(x) = max(x, 0).

			Arguments:
				data : np.ndarray : An n dimensional numpy array of data that the function will be mapping to an output.

			Return:
				output : np.ndarray : An n dimensional numpy array of the mapped data.

		"""

		return np.where(data>=0, data, 0)


	@accepts(self="any", data=np.ndarray)
	def calculate_gradients(self, data) -> np.ndarray:
		"""
			Calculates the derivative of the activation function.
		
			Arguments:
				data : np.ndarray : An n dimensional numpy array of data that the derivative will be calculated W.R.T.

			Return:
				output : np.ndarray : An n dimensional numpy array of the calculated derivative.

		"""

		return np.where(data>=0, 1, 0)





class Leaky_ReLU(Base_Activation):
	@accepts(self="any", alpha=float)
	def __init__(self, alpha=1.0e-1) -> None:
		"""
			The Leaky_ReLU class is a suggested improvement of the ReLU function. 
			Commonly used for hidden layers.
			
			Arguments:
				alpha : float : Allows for flow of the data if it's less than zero, fixes the dying ReLU problem.

		"""

		self.alpha = alpha
		self.name = "leaky_relu"


	@accepts(self="any", data=np.ndarray)	
	def map_data(self, data) -> np.ndarray:
		"""
			Maps some data to an output with the form of f(x) = max(x, alpha*x).

			Arguments:
				data : np.ndarray : An n dimensional numpy array of data that the function will be mapping to an output.

			Return:
				output : np.ndarray : An n dimensional numpy array of the mapped data.

		"""

		return np.where(data>=0, data, self.alpha*data)

	
	@accepts(self="any", data=np.ndarray)
	def calculate_gradients(self, data) -> np.ndarray:
		"""
			Calculates the derivative of the activation function.
		
			Arguments:
				data : np.ndarray : An n dimensional numpy array of data that the derivative will be calculated W.R.T.

			Return:
				output : np.ndarray : An n dimensional numpy array of the calculated derivative.

		"""

		return np.where(data>=0, 1, self.alpha)





class ELU(Base_Activation):
	@accepts(self="any", alpha=float)
	def __init__(self, alpha=1.0e-1) -> None:
		"""
			The ELU class is a suggested improvement of the ReLU function. 
			Commonly used for hidden layers.
			
			Arguments:
				alpha : float : Allows for flow of the data if it's less than zero, fixes the dying ReLU problem.

		"""

		self.alpha = alpha
		self.name = "elu"


	@accepts(self="any", data=np.ndarray)
	def map_data(self, data) -> np.ndarray:
		"""
			Maps some data to an output with the form of f(x) = max(x, alpha*(e^x - 1)).

			Arguments:
				data : np.ndarray : An n dimensional numpy array of data that the function will be mapping to an output.

			Return:
				output : np.ndarray : An n dimensional numpy array of the mapped data.

		"""

		return np.where(data>=0, data, self.alpha*(np.exp(data)-1))

	
	@accepts(self="any", data=np.ndarray)
	def calculate_gradients(self, data) -> np.ndarray:
		"""
			Calculates the derivative of the activation function.
		
			Arguments:
				data : np.ndarray : An n dimensional numpy array of data that the derivative will be calculated W.R.T.

			Return:
				output : np.ndarray : An n dimensional numpy array of the calculated derivative.

		"""

		return np.where(data>=0, 1, self.alpha*np.exp(data))





@accepts(activation=(Base_Activation, str))
def get(activation) -> Base_Activation:
	"""
		Finds and returns the correct activation function.

		Arguments:
			activation : Base_Activation/str : The activation function the user wants to use.

		Returns:
			activation : Base_Activation : The correct optimization function.
		
	"""

	if type(activation) == str:
		if activation.lower() in ("linear"):
			return Linear()
		elif activation.lower() in ("softmax"):
			return Softmax()
		elif activation.lower() in ("sigmoid"):
			return Sigmoid()
		elif activation.lower() in ("tanh"):
			return Tanh()
		elif activation.lower() in ("relu"):
			return ReLU()
		elif activation.lower() in ("leaky_relu", "lrelu"):
			return Leaky_ReLU()
		elif activation.lower() in ("elu"):
			return ELU()
		else:
			print("At activations.get(): '%s' is not an available activation function. Has been set to 'Linear' by default" % activation)
			return Linear()
	elif isinstance(activation, Base_Activation):
		return activation
	else:
		raise ValueError("At activations.get(): Expected 'class inheriting from Base_Activation' or 'str' for the argument 'activation', recieved '%s'" % type(activation))
