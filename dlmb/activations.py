from abc import ABCMeta, abstractmethod
import numpy as np

class Base_activation(metaclass=ABCMeta):
	@abstractmethod
	def __init__(self):
		"""
		The Base_activation class is an abstract class and makes sure every activation function uses the functions down below.

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
		Calculates the derivative of the activation function.
	
		Arguments:
			data - type = Numpy array: The data that the derivative will be calculate with respect to.

		Return:
			output - type = Numpy array: The calculated derivative.

		"""

		return output

# ------------------------------------------------------------------------------------------------------------------------
class Linear(Base_activation):
	def __init__(self):
		"""
		The Linear class is the default activation function and generally isn't actually used.

		"""

		self.name = "linear"

	# ------------------------------------------------------------------------------------------------------------------------
	def function(self, data):
		"""
		Maps some data to an output with the form of f(x) = x.

		Arguments:
			data - type = Numpy array: The data that the function will be mapping to an output.

		Return:
			output - type = Numpy array: The mapped data.

		"""

		return data

	# ------------------------------------------------------------------------------------------------------------------------
	def derivative(self, data):
		"""
		Calculates the derivative of the activation function.
	
		Arguments:
			data - type = Numpy array: The data that the derivative will be calculate with respect to.

		Return:
			output - type = Numpy array: The calculated derivative.

		"""

		return np.array([[1]])

# ------------------------------------------------------------------------------------------------------------------------
class Softmax(Base_activation):
	def __init__(self):
		"""
		The Softmax class takes an array and normalizes it into a probability distribution with the same size. Generally used for the output layer.

		"""

		self.name = "softmax"

	# ------------------------------------------------------------------------------------------------------------------------
	def function(self, data):
		"""
		Maps some data to an output with the form of f(x) = e^x_k / sum(e^x_i).

		Arguments:
			data - type = Numpy array: The data that the function will be mapping to an output.

		Return:
			output - type = Numpy array: The mapped data.

		"""

		e_x = np.exp(data-np.max(data))
		return e_x/(np.sum(e_x, axis=1, keepdims=True)+1.0e-8)

	# ------------------------------------------------------------------------------------------------------------------------
	def derivative(self, data):
		"""
		Calculates the derivative of the activation function.
	
		Arguments:
			data - type = Numpy array: The data that the derivative will be calculate with respect to.

		Return:
			output - type = Numpy array: The calculated derivative.

		"""

		a = np.reshape(self.function(data), (data.shape[0], data.shape[1], 1))
		e = np.ones((a.shape[1], 1))
		i = np.identity(a.shape[1])
		return (a*e.T) * (i - e*a.reshape((a.shape[0], a.shape[2], a.shape[1])))

# ------------------------------------------------------------------------------------------------------------------------
class Sigmoid(Base_activation):
	def __init__(self):
		"""
		The Sigmoid class squashes some data between 0 and 1. Good for probabilities and generally used for any layer.

		"""

		self.name = "sigmoid"

	# ------------------------------------------------------------------------------------------------------------------------
	def function(self, data):
		"""
		Maps some data to an output with the form of f(x) = 1/(1+e^-x).

		Arguments:
			data - type = Numpy array: The data that the function will be mapping to an output.

		Return:
			output - type = Numpy array: The mapped data.

		"""

		return 1/(1+np.exp(-data))

	# ------------------------------------------------------------------------------------------------------------------------
	def derivative(self, data):
		"""
		Calculates the derivative of the activation function.
	
		Arguments:
			data - type = Numpy array: The data that the derivative will be calculate with respect to.

		Return:
			output - type = Numpy array: The calculated derivative.

		"""

		return self.function(data) * (1-self.function(data))

# ------------------------------------------------------------------------------------------------------------------------
class Tanh(Base_activation):
	def __init__(self):
		"""
		The Tanh class is the hyperbolic tangent function. Squashes some data between -1 and 1.

		"""

		self.name = "tanh"

	# ------------------------------------------------------------------------------------------------------------------------
	def function(self, data):
		"""
		Maps some data to an output with the form of f(x) = sinh(x)/cosh(x).

		Arguments:
			data - type = Numpy array: The data that the function will be mapping to an output.

		Return:
			output - type = Numpy array: The mapped data.

		"""

		return np.tanh(data)

	# ------------------------------------------------------------------------------------------------------------------------
	def derivative(self, data):
		"""
		Calculates the derivative of the activation function.
	
		Arguments:
			data - type = Numpy array: The data that the derivative will be calculate with respect to.

		Return:
			output - type = Numpy array: The calculated derivative.

		"""

		return 1-self.function(data)**2

# ------------------------------------------------------------------------------------------------------------------------
class ReLU(Base_activation):
	def __init__(self):
		"""
		The ReLU class is the Rectified Linear Unit function. Commonly used for hidden layers.

		"""

		self.name = "relu"

	# ------------------------------------------------------------------------------------------------------------------------
	def function(self, data):
		"""
		Maps some data to an output with the form of f(x) = max(x, 0).

		Arguments:
			data - type = Numpy array: The data that the function will be mapping to an output.

		Return:
			output - type = Numpy array: The mapped data.

		"""

		return np.where(data>=0, data, 0)

	# ------------------------------------------------------------------------------------------------------------------------
	def derivative(self, data):
		"""
		Calculates the derivative of the activation function.
	
		Arguments:
			data - type = Numpy array: The data that the derivative will be calculate with respect to.

		Return:
			output - type = Numpy array: The calculated derivative.

		"""

		return np.where(data>=0, 1, 0)

# ------------------------------------------------------------------------------------------------------------------------
class Leaky_ReLU(Base_activation):
	def __init__(self, alpha=1.0e-1):
		"""
		The Leaky_ReLU class is a suggested improved version of the ReLU function. Commonly used for hidden layers.
		
		Arguments:
			alpha - type = float: A small number which adds some flow for the data if it's less than zero, fixes the dying ReLU problem.

		"""

		self.alpha = alpha
		self.name = "leaky_relu"

	# ------------------------------------------------------------------------------------------------------------------------
	def function(self, data):
		"""
		Maps some data to an output with the form of f(x) = max(x, alpha*x).

		Arguments:
			data - type = Numpy array: The data that the function will be mapping to an output.

		Return:
			output - type = Numpy array: The mapped data.

		"""

		return np.where(data>=0, data, alpha*data)

	# ------------------------------------------------------------------------------------------------------------------------
	def derivative(self, data):
		"""
		Calculates the derivative of the activation function.
	
		Arguments:
			data - type = Numpy array: The data that the derivative will be calculate with respect to.

		Return:
			output - type = Numpy array: The calculated derivative.

		"""

		return np.where(data>=0, 1, alpha)

# ------------------------------------------------------------------------------------------------------------------------
class ELU(Base_activation):
	def __init__(self, alpha=1.0e-1):
		"""
		The ELU class is a suggested improved version of the ReLU function. Commonly used for hidden layers.
		
		Arguments:
			alpha - type = float: A small number which adds some flow for the data if it's less than zero, fixes the dying ReLU problem.

		"""

		self.alpha = alpha
		self.name = "elu"

	# ------------------------------------------------------------------------------------------------------------------------
	def function(self, data):
		"""
		Maps some data to an output with the form of f(x) = max(x, alpha*(e^x - 1)).

		Arguments:
			data - type = Numpy array: The data that the function will be mapping to an output.

		Return:
			output - type = Numpy array: The mapped data.

		"""

		return np.where(data>=0, data, self.alpha*(np.exp(data)-1))

	# ------------------------------------------------------------------------------------------------------------------------
	def derivative(self, data):
		"""
		Calculates the derivative of the activation function.
	
		Arguments:
			data - type = Numpy array: The data that the derivative will be calculate with respect to.

		Return:
			output - type = Numpy array: The calculated derivative.

		"""

		return np.where(data>=0, 1, self.alpha*np.exp(data))