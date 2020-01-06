from abc import ABCMeta, abstractmethod
import numpy as np

from utils.function_helpers import *


# For all classes optimizations[0] is the calculated gradients W.R.T the vars from the layer, and anything other than that is just optimizations from the optimizer function.
class Base_Optimizer(metaclass=ABCMeta):
	@abstractmethod
	def __init__(self) -> None:
		"""
			The Base_Optimizer class is an abstract class for all optimizer functions. 
			All optimizer functions must inherit from Base_Optimizer.

		"""

		pass


	@abstractmethod
	def map_data(self, optimizations:list, epoch:int) -> list:

		"""
			The optimizer takes some data and applies a mathematical mapping to it.
		
			Arguments:
				optimizations : listnp.ndarrayY] : A list containing:
					gradients     : np.ndarray : An n dimensional numpy array of gradients for the vars that are being updated.
					optimizations : np.ndarray : An n dimensional numpy array of optimizations from the previous batch.

				epoch : int : The current epoch or time step the model is at.

			Returns:
				output : list : A list containing:
					updates       : np.ndarray : An n dimensional numpy array of the optimized gradients that will be used to update the vars.
					optimizations : np.ndarray : An n dimensional numpy array of optimizations from this batch.

		"""

		return output





class GD(Base_Optimizer):
	@accepts(self="any", lr=float, momentum=float)
	def __init__(self, lr=0.1, momentum=0.9) -> None:
		"""
			GD (Gradient Decent) is the simplest of all the optimizers. 
			It applies some momentum to the gradients of the vars that are being updated.
			It is reccomended to leave the arguments as they are.

			Arguments:
				lr       : float : Controls how fast the model learns, however choosing a big or small lr isn't always good.
				momentum : float : Controls the speed of convergence to a local minima, however choosing a big or small momentum isn't always good.

		"""

		self.lr = lr
		self.momentum = momentum


	@accepts(self="any", optimizations=list, epoch=int)
	def map_data(self, optimizations, epoch) -> list:

		"""
			Applies some momentum to the gradients so convergence to the local minima of the loss function will be quicker.

			Arguments:
				optimizations : list : A list containing:
					gradients     : np.ndarray : An n dimensional numpy array of gradients for the vars that are being updated.
					optimizations : np.ndarray : An n dimensional numpy array of optimizations from the previous batch.

				epoch : int : The current epoch or time step the model is at.

			Returns:
				output : list : A list containing:
					updates       : np.ndarray : An n dimensional numpy array of the optimized gradients that will be used to update the vars.
					optimizations : np.ndarray : An n dimensional numpy array of optimizations from this batch.

		"""

		optimizations[1] = self.momentum*optimizations[1] + (1-self.momentum)*optimizations[0]
		return [self.lr*optimizations[1], optimizations[1]]





class RMSprop(Base_Optimizer):
	@accepts(self="any", lr=float, beta=float)
	def __init__(self, lr=0.001, beta=0.99) -> None:
		"""
			RMSprop is usually a good choice for recurrent neural networks.
			It is reccomended to leave the arguments as they are.

			Arguments:
				lr   : float : Controls how fast the model learns, however choosing a big or small lr isn't always good.
				beta : float : Helps the gradients to find the optimal direction to head in.

		"""

		self.lr = lr
		self.beta = beta


	@accepts(self="any", optimizations=list, epoch=int)
	def map_data(self, optimizations, epoch) -> list:

		"""
			Applies a momentum like operation to the gradients so the gradients can find the optimal direction to head in.

			Arguments:
				optimizations : list : A list containing:
					gradients     : np.ndarray : An n dimensional numpy array of gradients for the vars that are being updated.
					optimizations : np.ndarray : An n dimensional numpy array of optimizations from the previous batch.

				epoch : int : The current epoch or time step the model is at.

			Returns:
				output : list : A list containing:
					updates       : np.ndarray : An n dimensional numpy array of the optimized gradients that will be used to update the vars.
					optimizations : np.ndarray : An n dimensional numpy array of optimizations from this batch.

		"""

		optimizations[1] = self.beta*optimizations[1] + (1-self.beta) * optimizations[0]**2
		return [division_check(self.lr, np.sqrt(optimizations[1])) * optimizations[0], optimizations[1]]





class Adagrad(Base_Optimizer):
	@accepts(self="any", lr=float)
	def __init__(self, lr=0.001) -> None:
		"""
			Adagrad is very similar to RMSprop.
			It is reccomended to leave the arguments as they are.

			Arguments:
				lr   : float : Controls how fast the model learns, however choosing a big or small lr isn't always good.

		"""

		self.lr = lr


	@accepts(self="any", optimizations=list, epoch=int)
	def map_data(self, optimizations, epoch) -> list:

		"""
			Adds element-wise scaling of the gradient based on the previous sum of squares.

			Arguments:
				optimizations : list : A list containing:
					gradients     : np.ndarray : An n dimensional numpy array of gradients for the vars that are being updated.
					optimizations : np.ndarray : An n dimensional numpy array of optimizations from the previous batch.

				epoch : int : The current epoch or time step the model is at.

			Returns:
				output : list : A list containing:
					updates       : np.ndarray : An n dimensional numpy array of the optimized gradients that will be used to update the vars.
					optimizations : np.ndarray : An n dimensional numpy array of optimizations from this batch.

		"""

		optimizations[1] += optimizations[0]**2
		return [division_check(self.lr, np.sqrt(optimizations[1])) * optimizations[0], optimizations[1]]





class Adam(Base_Optimizer):
	@accepts(self="any", lr=float, momentum=float, beta=float)
	def __init__(self, lr=0.01, momentum=0.9, beta=0.99) -> None:
		"""
			Adam is generally the most widly used optimizer.
			It is reccomended to leave the arguments as they are.

			Arguments:
				lr       : float : Controls how fast the model learns, however choosing a big or small lr isn't always good.
				momentum : float : Controls the speed of convergence to a local minima, however choosing a big or small momentum isn't always good.
				beta : float : Helps the gradients to find the optimal direction to head in.

		"""

		self.lr = lr
		self.momentum = momentum
		self.beta = beta


	@accepts(self="any", optimizations=list, epoch=int)
	def map_data(self, optimizations, epoch) -> list:

		"""
			Applies momentum and a momentum like operation to the gradients so the gradients can find the optimal direction to head in.

			Arguments:
				optimizations : list : A list containing:
					gradients   : np.ndarray : An n dimensional numpy array of gradients for the vars that are being updated.
					momentum_op : np.ndarray : An n dimensional numpy array of momentum optimizations from the previous batch.
					beta_op     : np.ndarray : An n dimensional numpy array of beta optimizations from the previous batch.

				epoch : int : The current epoch or time step the model is at.

			Returns:
				output : list : A list containing:
					updates     : np.ndarray : An n dimensional numpy array of the optimized gradients that will be used to update the vars.
					momentum_op : np.ndarray : An n dimensional numpy array of momentum optimizations from this batch.
					beta_op : np.ndarray : An n dimensional numpy array of beta optimizations from this batch.

		"""

		# Checks if the beta optimizations aren't there, meaning this is the first time the optimizer is getting used within this train loop.
		if len(optimizations) < 3:
			optimizations.append(0)

		optimizations[1] = self.momentum*optimizations[1] + (1-self.momentum)*optimizations[0]
		optimizations[2] = self.beta*optimizations[2] + (1-self.beta)*optimizations[0]**2

		momentum_hat = division_check(optimizations[1], 1-self.momentum**epoch)
		beta_hat = division_check(optimizations[2], 1-self.beta**epoch)

		return [division_check(self.lr * momentum_hat, np.sqrt(beta_hat)+1.0e-8), optimizations[1], optimizations[2]]





class Adamax(Base_Optimizer):
	@accepts(self="any", lr=float, momentum=float, beta=float)
	def __init__(self, lr=0.01, momentum=0.9, beta=0.99) -> None:
		"""
			Adamax is very similar to Adam.
			It is reccomended to leave the arguments as they are.

			Arguments:
				lr       : float : Controls how fast the model learns, however choosing a big or small lr isn't always good.
				momentum : float : Controls the speed of convergence to a local minima, however choosing a big or small momentum isn't always good.
				beta : float : Helps the gradients to find the optimal direction to head in.

		"""

		self.lr = lr
		self.momentum = momentum
		self.beta = beta


	@accepts(self="any", optimizations=list, epoch=int)
	def map_data(self, optimizations, epoch) -> list:

		"""
			Applies momentum and a momentum like operation to the gradients so the gradients can find the optimal direction to head in.

			Arguments:
				optimizations : list : A list containing:
					gradients   : np.ndarray : An n dimensional numpy array of gradients for the vars that are being updated.
					momentum_op : np.ndarray : An n dimensional numpy array of momentum optimizations from the previous batch.
					beta_op     : np.ndarray : An n dimensional numpy array of beta optimizations from the previous batch.

				epoch : int : The current epoch or time step the model is at.

			Returns:
				output : list : A list containing:
					updates     : np.ndarray : An n dimensional numpy array of the optimized gradients that will be used to update the vars.
					momentum_op : np.ndarray : An n dimensional numpy array of momentum optimizations from this batch.
					beta_op : np.ndarray : An n dimensional numpy array of beta optimizations from this batch.

		"""

		# Checks if the beta optimizations aren't there, meaning this is the first time the optimizer is getting used within this train loop.
		if len(optimizations) < 3:
			optimizations.append(0)

		optimizations[1] = self.momentum*optimizations[1] + (1-self.momentum)*optimizations[0]
		optimizations[2] = np.maximum(self.beta*optimizations[2], np.abs(optimizations[0]))

		momentum_hat = division_check(optimizations[1], 1-self.momentum**epoch)

		return [division_check(self.lr * momentum_hat, self.beta), optimizations[1], optimizations[2]]





class Nadam(Base_Optimizer):
	@accepts(self="any", lr=float, momentum=float, beta=float)
	def __init__(self, lr=0.01, momentum=0.9, beta=0.99) -> None:
		"""
			Nadam is very similar to Adam.
			It is reccomended to leave the arguments as they are.

			Arguments:
				lr       : float : Controls how fast the model learns, however choosing a big or small lr isn't always good.
				momentum : float : Controls the speed of convergence to a local minima, however choosing a big or small momentum isn't always good.
				beta : float : Helps the gradients to find the optimal direction to head in.

		"""

		self.lr = lr
		self.momentum = momentum
		self.beta = beta


	@accepts(self="any", optimizations=list, epoch=int)
	def map_data(self, optimizations, epoch) -> list:

		"""
			Applies momentum and a momentum like operation to the gradients so the gradients can find the optimal direction to head in.

			Arguments:
				optimizations : list : A list containing:
					gradients   : np.ndarray : An n dimensional numpy array of gradients for the vars that are being updated.
					momentum_op : np.ndarray : An n dimensional numpy array of momentum optimizations from the previous batch.
					beta_op     : np.ndarray : An n dimensional numpy array of beta optimizations from the previous batch.

				epoch : int : The current epoch or time step the model is at.

			Returns:
				output : list : A list containing:
					updates     : np.ndarray : An n dimensional numpy array of the optimized gradients that will be used to update the vars.
					momentum_op : np.ndarray : An n dimensional numpy array of momentum optimizations from this batch.
					beta_op : np.ndarray : An n dimensional numpy array of beta optimizations from this batch.

		"""

		# Checks if the beta optimizations aren't there, meaning this is the first time the optimizer is getting used within this train loop.
		if len(optimizations) < 3:
			optimizations.append(0)

		optimizations[1] = self.momentum*optimizations[1] + (1-self.momentum)*optimizations[0]
		optimizations[2] = np.maximum(self.beta*optimizations[2], np.abs(optimizations[0]))

		momentum_hat = division_check(optimizations[1], 1-self.momentum**epoch) + division_check((1 - self.momentum) * optimizations[0], 1 - self.momentum**epoch)
		beta_hat = division_check(optimizations[2], 1-self.beta**epoch) 

		return [division_check(self.lr * momentum_hat, np.sqrt(beta_hat)+1.0e-8), optimizations[1], optimizations[2]]





@accepts(optimizer=(Base_Optimizer, str))
def get(optimizer) -> Base_Optimizer:
	"""
		Finds and returns the correct optimizer function.

		Arguments:
			optimizer : OPTIMIZER/str : The optimization function the user wants to use.

		Returns:
			optimizer : OPTIMIZER : The correct optimization function.
		
	"""

	if type(optimizer) == str:
		if optimizer.lower() in ("gd", 'gradient_decent'):
			return GD()
		elif optimizer.lower() in ("rmsprop"):
			return RMSprop()
		elif optimizer.lower() in ("adam"):
			return Adam()
		elif optimizer.lower() in ("adagrad"):
			return Adagrad()
		elif optimizer.lower() in ("adamax"):
			return Adamax()
		elif optimizer.lower() in ("nadam"):
			return Nadam()
		else:
			print("At optimizers.get(): '%s' is not an available optimizer function. Has been set to 'GD' by default" % optimizer)
			return GD()
	elif isinstance(optimizer, Base_Optimizer):
		return optimizer
	else:
		raise ValueError("At optimizers.get(): Expected 'class inheriting from Base_Optimizer' or 'str' for the argument 'optimizer', recieved '%s'" % type(optimizer))

