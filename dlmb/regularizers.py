import numpy as np
from abc import ABCMeta, abstractmethod

class base_regularization(metaclass=ABCMeta):
	@abstractmethod
	def __init__(self):
		"""
		The regularization classes help prevent over fitting by adding a penalty to the vars.

		"""

		pass

	# ------------------------------------------------------------------------------------------------------------------------
	@abstractmethod
	def regularize(self, variables):
		"""
		Takes some variables and calculates a penalty for them.

		Arguments:
			variables - type = Numpy array: A numpy array of the variables that will be regularized.

		Returns:
			output - type = Numpy array: The calculated penalties for the variables.

		"""

		return output

# ------------------------------------------------------------------------------------------------------------------------
class L0:
	def __init__(self):
		"""
		L0 is just a default class (like the Linear class in activations.py).

		"""

		self.name = "l0"

	# ------------------------------------------------------------------------------------------------------------------------
	def regularize(self, variables):
		"""
		Takes some variables and calculates a penalty for them.

		Arguments:
			variables - type = Numpy array: A numpy array of the variables that will be regularized.

		Returns:
			output - type = Numpy array: The calculated penalties for the variables.

		"""

		return 0 

# ------------------------------------------------------------------------------------------------------------------------
class L1:
	def __init__(self, penalty=0.03):
		"""
		L1 is like L2 but sometimes the regularized variables can reach zero.

		"""

		self.penalty = penalty
		self.name = "l1"

	# ------------------------------------------------------------------------------------------------------------------------
	def regularize(self, variables):
		"""
		Takes some variables and calculates a penalty for them.

		Arguments:
			variables - type = Numpy array: A numpy array of the variables that will be regularized.

		Returns:
			output - type = Numpy array: The calculated penalties for the variables.

		"""

		return self.penalty *np.sign(variables)

# ------------------------------------------------------------------------------------------------------------------------
class L2:
	def __init__(self, penalty=0.03):
		"""
		L2 is the most commonly used regularization technique out of all of them. L2 trys to minimalize the variables but never reaches zero.
	
		"""

		self.penalty = penalty
		self.name = "l2"

	# ------------------------------------------------------------------------------------------------------------------------
	def regularize(self, variables):
		"""
		Takes some variables and calculates a penalty for them.

		Arguments:
			variables - type = Numpy array: A numpy array of the variables that will be regularized.

		Returns:
			output - type = Numpy array: The calculated penalties for the variables.

		"""

		return self.penalty * variables