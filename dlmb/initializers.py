import numpy as np

def zeros(matrix_shape):
	"""
	Creates a numpy array of shape matrix_shape filled with values of 0.

	Arguments:
		matrix_shape - tuple: A tuple with 2 numbers, specifying size of the numpy array.

	Returns:
		output - Numpy array: A numpy array filled with values of 0.
		
	"""

	return np.zeros(matrix_shape)

# ------------------------------------------------------------------------------------------------------------------------
def ones(matrix_shape):
	"""
	Creates a numpy array of shape matrix_shape filled with values of 1.

	Arguments:
		matrix_shape - tuple: A tuple with 2 numbers, specifying size of the numpy array.

	Returns:
		output - Numpy array: A numpy array filled with values of 1.
		
	"""

	return np.ones(matrix_shape)

# ------------------------------------------------------------------------------------------------------------------------
def random(matrix_shape):
	"""
	Creates a numpy array of shape matrix_shape filled with random values between 0 and 1.

	Arguments:
		matrix_shape - tuple: A tuple with 2 numbers, specifying size of the numpy array.

	Returns:
		output - Numpy array: A numpy array filled with random values between 0 and 1.
		
	"""

	return np.random.random(matrix_shape)

# ------------------------------------------------------------------------------------------------------------------------
def uniform(matrix_shape):
	"""
	Creates a uniform numpy array with boundaries of [-1/sqrt(n), 1/sqrt(n)].

	Arguments:
		matrix_shape - tuple: A tuple with 2 numbers, specifying size of the numpy array.

	Returns:
		output - Numpy array: A uniform numpy array.
		
	"""

	boundaries = 1/np.sqrt(matrix_shape[1])
	return np.random.uniform(-boundaries, boundaries, matrix_shape)

# ------------------------------------------------------------------------------------------------------------------------
def xavier(matrix_shape):
	"""
	Creates a gaussian distribution numpy array with a mean of 0 and variance of sqrt(2/(n+m)).

	Arguments:
		matrix_shape - tuple: A tuple with 2 numbers, specifying size of the numpy array.

	Returns:
		output - Numpy array: A gaussian distribution numpy array.
		
	"""

	return np.random.normal(0, np.sqrt(2/(matrix_shape[0]+matrix_shape[1])), matrix_shape)

# ------------------------------------------------------------------------------------------------------------------------
def xavier_uniform(matrix_shape):
	"""
	Creates a uniform numpy array based of the xavier initializer with boundaries of [-sqrt(6)/sqrt(n+m), sqrt(6)/sqrt(n+m)].

	Arguments:
		matrix_shape - tuple: A tuple with 2 numbers, specifying size of the numpy array.

	Returns:
		output - Numpy array: A uniform numpy array.
		
	"""

	boundaries = np.sqrt(6)/np.sqrt(matrix_shape[0]+matrix_shape[1])
	return np.random.uniform(-boundaries, boundaries, matrix_shape)

# ------------------------------------------------------------------------------------------------------------------------
def sigmoid_uniform(matrix_shape):
	"""
	Creates a uniform numpy array based of the xavier_uniform initializer with boundaries of [-(4*sqrt(6))/sqrt(n+m), 4*sqrt(6)/sqrt(n+m)].

	Arguments:
		matrix_shape - tuple: A tuple with 2 numbers, specifying size of the numpy array.

	Returns:
		output - Numpy array: A uniform numpy array.
		
	"""

	boundaries = 4*np.sqrt(6)/np.sqrt(matrix_shape[0]+matrix_shape[1])
	return np.random.uniform(-boundaries, boundaries, matrix_shape)

# ------------------------------------------------------------------------------------------------------------------------
def relu(matrix_shape):
	"""
	Creates a gaussian distribution numpy array with a mean of 0 and variance of sqrt(2/m).

	Arguments:
		matrix_shape - tuple: A tuple with 2 numbers, specifying size of the numpy array.

	Returns:
		output - Numpy array: A gaussian distribution numpy array.
		
	"""

	return np.random.normal(0, np.sqrt(2/matrix_shape[0]), matrix_shape)

# ------------------------------------------------------------------------------------------------------------------------
def relu_uniform(matrix_shape):
	"""
	Creates a uniform numpy array based of the relu initializer with boundaries of [-sqrt(6/m), sqrt(6/m)].

	Arguments:
		matrix_shape - tuple: A tuple with 2 numbers, specifying size of the numpy array.

	Returns:
		output - Numpy array: A uniform numpy array.
		
	"""

	boundaries = np.sqrt(6/matrix_shape[0])
	return np.random.uniform(-boundaries, boundaries, matrix_shape)

# ------------------------------------------------------------------------------------------------------------------------
def get(initializer):
	"""
	Finds and returns the correct initializer function.

	Arguments:
		initializer - str/instance of a function: The initializer function.

	Returns:
		initializer - instance of a function: The correct initializer function.
		
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
		elif initializer.lower() in ("xavier"):
			return xavier
		elif initializer.lower() in ("xavier_uniform"):
			return xavier_uniform
		elif initializer.lower() in ("sigmoid_uniform"):
			return sigmoid_uniform
		elif initializer.lower() in ("relu"):
			return relu
		elif initializer.lower() in ("relu_uniform"):
			return relu_uniform
		else:
			print("'%s' is not currently an available initializer function. Has been set to 'random' by default" % initializer)
			return random
	else:
		return initializer
