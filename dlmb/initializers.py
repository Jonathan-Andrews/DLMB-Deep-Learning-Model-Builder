import numpy as np
from utils.function_helpers import *

@accepts(shape=tuple)
def zeros(shape) -> np.ndarray:
	"""
		Creates a numpy array of shape shape filled with values of 0.

		Arguments:
			shape : tuple : A tuple with 2 numbers, specifying size of the numpy array.

		Returns:
			output : np.ndarray : A numpy array filled with values of 0.
		
	"""

	return np.zeros(shape)


@accepts(shape=tuple)
def ones(shape) -> np.ndarray:
	"""
		Creates a numpy array of shape shape filled with values of 1.

		Arguments:
			shape : tuple : A tuple with 2 numbers, specifying size of the numpy array.

		Returns:
			output : Numpy array : A numpy array filled with values of 1.
		
	"""

	return np.ones(shape)


@accepts(shape=tuple)
def random(shape) -> np.ndarray:
	"""
		Creates a numpy array of shape shape filled with random values between 0 and 1.

		Arguments:
			shape : tuple : A tuple with 2 numbers, specifying size of the numpy array.

		Returns:
			output : np.ndarray : A numpy array filled with random values between 0 and 1.
		
	"""

	return np.random.random(shape)


@accepts(shape=tuple)
def uniform(shape) -> np.ndarray:
	"""
		Creates a uniform numpy array with boundaries of [-1/sqrt(n), 1/sqrt(n)].

		Arguments:
			shape : tuple : A tuple with 2 numbers, specifying size of the numpy array.

		Returns:
			output : np.ndarray : A uniform numpy array.
		
	"""

	boundaries = 1/np.sqrt(shape[1])
	return np.random.uniform(-boundaries, boundaries, shape)


@accepts(shape=tuple)
def xavier(shape) -> np.ndarray:
	"""
		Creates a gaussian distribution numpy array with a mean of 0 and variance of sqrt(2/(n+m)).

		Arguments:
			shape : tuple : A tuple with 2 numbers, specifying size of the numpy array.

		Returns:
			output : np.ndarray : A uniform numpy array.
		
	"""

	return np.random.normal(0, np.sqrt(2/(shape[0]+shape[1])), shape)


@accepts(shape=tuple)
def xavier_uniform(shape) -> np.ndarray:
	"""
		Creates a uniform numpy array based of the xavier initializer with boundaries of [-sqrt(6)/sqrt(n+m), sqrt(6)/sqrt(n+m)].

		Arguments:
			shape : tuple : A tuple with 2 numbers, specifying size of the numpy array.

		Returns:
			output : np.ndarray : A uniform numpy array.
		
	"""

	boundaries = np.sqrt(6)/np.sqrt(shape[0]+shape[1])
	return np.random.uniform(-boundaries, boundaries, shape)


@accepts(shape=tuple)
def sigmoid_uniform(shape) -> np.ndarray:
	"""
		Creates a uniform numpy array based of the xavier_uniform initializer with boundaries of [-(4*sqrt(6))/sqrt(n+m), 4*sqrt(6)/sqrt(n+m)].

		Arguments:
			shape : tuple : A tuple with 2 numbers, specifying size of the numpy array.

		Returns:
			output : np.ndarray : A uniform numpy array.
		
	"""

	boundaries = 4*np.sqrt(6)/np.sqrt(shape[0]+shape[1])
	return np.random.uniform(-boundaries, boundaries, shape)


@accepts(shape=tuple)
def relu(shape) -> np.ndarray:
	"""
		Creates a gaussian distribution numpy array with a mean of 0 and variance of sqrt(2/m).

		Arguments:
			shape : tuple : A tuple with 2 numbers, specifying size of the numpy array.

		Returns:
			output : np.ndarray : A uniform numpy array.
		
	"""

	return np.random.normal(0, np.sqrt(2/shape[1]), shape)


@accepts(shape=tuple)
def relu_uniform(shape):
	"""
		Creates a uniform numpy array based of the relu initializer with boundaries of [-sqrt(6/m), sqrt(6/m)].
		
		Arguments:
			shape : tuple : A tuple with 2 numbers, specifying size of the numpy array.

		Returns:
			output : np.ndarray : A uniform numpy array.
		
	"""

	boundaries = np.sqrt(6/shape[1])
	return np.random.uniform(-boundaries, boundaries, shape)


def get(initializer):
	"""
		Finds and returns the correct initializer function.

		Arguments:
			initializer : str/callable : The initializer function.

		Returns:
			initializer : callable : The correct initializer function.
		
	"""

	if type(initializer) == str:
		if initializer.lower() in ("zeros", "zero"):
			return zeros
		elif initializer.lower() in ("ones", "one"):
			return ones
		elif initializer.lower() in ("random"):
			return random
		elif initializer.lower() in ("uniform"):
			return uniform
		elif initializer.lower() in ("xavier", "glorot"):
			return xavier
		elif initializer.lower() in ("xavier_uniform", "glorot_uniform"):
			return xavier_uniform
		elif initializer.lower() in ("sigmoid_uniform"):
			return sigmoid_uniform
		elif initializer.lower() in ("relu"):
			return relu
		elif initializer.lower() in ("relu_uniform"):
			return relu_uniform
		else:
			print("'%s' is not currently an available initializer function. Has been set to 'uniform' by default" % initializer)
			return uniform
	else:
		return initializer
