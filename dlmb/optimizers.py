from abc import ABCMeta, abstractmethod
import numpy as np

# For all classes optimizations[0] is the calculated gradients and anything other than that is just optimizations from the optimizer class

class Base_optimizer(metaclass=ABCMeta):
	@abstractmethod
	def __init__(self):
		"""
		The Base_optimizer class is an abstract class and makes sure every optimizer function uses the functions down below.

		"""

		pass

	# ------------------------------------------------------------------------------------------------------------------------
	@abstractmethod
	def optimize(self, optimizations, epoch):
		"""
		This is where the math happens. The optimizer takes some data and applies a mathematical mapping to it.
	
		Arguments:
			optimizations - type = Array: An array containing:
				gradients - type = Numpy array: A numpy array of gradients for the vars that are being updated.
				optimizations - type = Numpy array: A numpy array of optimizations from the previous time step.

		Returns:
			output - type = Array:  An array containing:
				updates - type = Numpy array: A numpy of the optimized gradients that will be used to update the vars.
				optimizations - type = Numpy array: A numpy array of optimizations from this time step.

		"""

		return output

# ------------------------------------------------------------------------------------------------------------------------
class GD(Base_optimizer):
	def __init__(self, lr=0.1, momentum=0.9):
		"""
		GD (Gradient Decent) is the simplest of all optimizers. 
		Applies some momentum to the gradients of the vars that are being updated.
		It is reccomended to leave the arguments as they are.

		Arguments:
			lr - type = float: A float that takes a small fraction of the gradients. Controls how fast the model learns, however choosing a big lr isn't always good.
			momentum - type = float: A float that controls the speed of convergence to a local minima, however choosing a big momentum isn't aslways good.

		"""

		self.lr = lr
		self.momentum = momentum

	# ------------------------------------------------------------------------------------------------------------------------
	def optimize(self, optimizations, epoch):
		"""
		Applies some momentum to the gradients so convergence to the local minima of the loss function will be quicker.

		Arguments:
			optimizations - type = Array: An array containing:
				gradients - type = Numpy array: A numpy array of gradients for the vars that are being updated.
				optimizations - type = Numpy array: A numpy array of optimizations from the previous time step.

		Returns:
			output - type = Array:  An array containing:
				updates - type = Numpy array: A numpy of the optimized gradients that will be used to update the vars.
				optimizations - type = Numpy array: A numpy array of optimizations from this time step.

		"""

		optimizations[1] = self.momentum*optimizations[1] + (1-self.momentum)*optimizations[0]
		return [self.lr*optimizations[1], optimizations[1]]

# ------------------------------------------------------------------------------------------------------------------------s
class RMSprop(Base_optimizer):
	def __init__(self, lr=0.001, beta=0.99, epsilon=1e-8):
		"""
		RMSprop is usually a good choice for recurrent neural networks.
		It is reccomended to leave the arguments as they are.

		Arguments:
			lr: A learning rate that takes a small fraction of the gradients. Controls how fast the model learns, however choosing a big lr isn't always good.
			beta: Helps the gradients find the optimal direction to head in.
			epsilon: A very small number that helps to stop division by zero.

		"""

		self.lr = lr
		self.beta = beta
		self.epsilon = epsilon

	# ------------------------------------------------------------------------------------------------------------------------
	def optimize(self, optimizations, epoch):
		"""
		Applies a momentum like operation to the gradients so the gradients can find the optimal direction to head in.

		Arguments:
			optimizations - type = Array: An array containing:
				gradients - type = Numpy array: A numpy array of gradients for the vars that are being updated.
				optimizations - type = Numpy array: A numpy array of optimizations from the previous time step.

		Returns:
			output - type = Array:  An array containing:
				updates - type = Numpy array: A numpy of the optimized gradients that will be used to update the vars.
				optimizations - type = Numpy array: A numpy array of optimizations from this time step.

		"""

		optimizations[1] = self.beta*optimizations[1] + (1-self.beta) * optimizations[0]**2
		return [(self.lr / (np.sqrt(optimizations[1])+self.epsilon)) * optimizations[0], optimizations[1]]

# ------------------------------------------------------------------------------------------------------------------------
class Adam(Base_optimizer):
	def __init__(self, lr=0.01, momentum=0.9, beta=0.99, epsilon=1e-8):
		"""
		Adam is generally the most usde optimizer.
		It is reccomended to leave the arguments as they are.

		Arguments:
			lr: A learning rate that takes a small fraction of the gradients. Controls how fast the model learns, however choosing a big lr isn't always good.
			momentum: How much momentum to add to the gradients.
			beta: Helps the gradients find the optimal direction to head in.
			epsilon: A very small number that helps to stop division by zero. 

		"""

		self.lr = lr
		self.momentum = momentum
		self.beta = beta
		self.epsilon = epsilon

	# ------------------------------------------------------------------------------------------------------------------------
	def optimize(self, optimizations, epoch):
		"""
		Applies momentum and a momentum like operation to the gradients so the gradients can find the optimal direction to head in.

		Arguments:
			optimizations - type = Array: An array containing:
				gradients - type = Numpy array: A numpy array of gradients for the vars that are being updated.
				beta - type = Numpy array: A numpy array of beta optimizations from the previous time step.
				momentum - type = Numpy array: A numpy array of momentum optimizations from the previous time step.

		Returns:
			output - type = Array:  An array containing:
				updates - type = Numpy array: A numpy of the optimized gradients that will be used to update the vars.
				beta - type = Numpy array: A numpy array of beta optimizations from this time step.
				momentum - type = Numpy array: A numpy array of momentum optimizations from this time step.

		"""

		# Checks if the momentum optimizations aren't there, meaning this is the first time the optimizer is getting used within this train loop
		if len(optimizations) < 3:
			optimizations.append(0)

		optimizations[1] = self.beta*optimizations[1] + (1-self.beta)*optimizations[0]**2
		optimizations[2] = self.momentum*optimizations[2] + (1-self.momentum)*optimizations[0]

		corrected_RMSprop = optimizations[1] / (1-self.beta**epoch)+self.epsilon
		corrected_momentum = optimizations[2] / (1-self.momentum**epoch)+self.epsilon

		return [self.lr * (corrected_momentum / (np.sqrt(corrected_RMSprop)+self.epsilon)), optimizations[1], optimizations[2]]