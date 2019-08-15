import numpy as np
from abc import ABCMeta, abstractmethod

class base_regularization(metaclass=ABCMeta):
	@abstractmethod
	def __init__(self):
		"""
		The regularization classes help prevent over-fitting by adding a penalty to the vars.

		"""

		pass

	# ------------------------------------------------------------------------------------------------------------------------
	@abstractmethod
	def map_data(self, variables):
		"""
		Takes some variables and calculates a penalty for them.

		Arguments:
			variables - Numpy array: A numpy array of the variables that will be regularized.

		Returns:
			output - Numpy array: The calculated penalties for the variables.

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
	def map_data(self, variables):
		"""
		Takes some variables and calculates a penalty for them.

		Arguments:
			variables - Numpy array: A numpy array of the variables that will be regularized.

		Returns:
			output - Numpy array: The calculated penalties for the variables.

		"""

		return 0 

# ------------------------------------------------------------------------------------------------------------------------
class L1:
	def __init__(self, penalty=0.03):
		"""
		L1 trys to minimize the variables, and may cause the variables to go to 0.

		"""

		self.penalty = penalty
		self.name = "l1"

	# ------------------------------------------------------------------------------------------------------------------------
	def map_data(self, variables):
		"""
		Takes some variables and calculates a penalty for them.

		Arguments:
			variables - Numpy array: A numpy array of the variables that will be regularized.

		Returns:
			output - Numpy array: The calculated penalties for the variables.

		"""

		return self.penalty *np.sign(variables)

# ------------------------------------------------------------------------------------------------------------------------
class L2:
	def __init__(self, penalty=0.03):
		"""
		L2 is the most commonly used regularization technique. L2 trys to minimize the variables but never reaches zero.
	
		"""

		self.penalty = penalty
		self.name = "l2"

	# ------------------------------------------------------------------------------------------------------------------------
	def map_data(self, variables):
		"""
		Takes some variables and calculates a penalty for them.

		Arguments:
			variables - Numpy array: A numpy array of the variables that will be regularized.

		Returns:
			output - Numpy array: The calculated penalties for the variables.

		"""

		return self.penalty * variables

# ------------------------------------------------------------------------------------------------------------------------
def get(regularizer):
	"""
	Finds and returns the correct regularizer class.

	Arguments:
		regularizer - str/instance of a class: The regularization class.

	Returns:
		regularizer - instance of a class: The correct regularization class.
		
	"""

	if type(regularizer) == str:
		if regularizer.lower() in ("l0"):
			return L0()
		elif regularizer.lower() in ("l1"):
			return L1()
		elif regularizer.lower() in ("l2"):
			return L2()
		else:
			print("'%s' is not currently an available regularizer. Has been set to 'L0' by default" % regularizer)
			return L0()
	else:
		return regularizer
