import numpy as np

def zeros(matrix_shape):
	"""
	Creates a matrix of shape matrix_shape filled with values of 0.

	Arguments:
		matrix_shape - type = tuple: A tuple with 2 numbers, specifying the number of rows and the number of columns.

	Returns:
		output - type = Numpy array: A matrix filled with random values between 0 and 1.
		
	"""

	return np.zeros(matrix_shape)

# ------------------------------------------------------------------------------------------------------------------------
def ones(matrix_shape):
	"""
	Creates a matrix of shape matrix_shape filled with values of 1.

	Arguments:
		matrix_shape - type = tuple: A tuple with 2 numbers, specifying the number of rows and the number of columns.

	Returns:
		output - type = Numpy array: A matrix filled with random values between 0 and 1.
		
	"""

	return np.ones(matrix_shape)

# ------------------------------------------------------------------------------------------------------------------------
def random(matrix_shape):
	"""
	Creates a matrix of shape matrix_shape filled with random values between 0 and 1.

	Arguments:
		matrix_shape - type = tuple: A tuple with 2 numbers, specifying the number of rows and the number of columns.

	Returns:
		output - type = Numpy array: A matrix filled with random values between 0 and 1.
		
	"""

	return np.random.random(matrix_shape)

# ------------------------------------------------------------------------------------------------------------------------
def uniform(matrix_shape):
	"""
	Creates a uniform matrix with boundaries of [-1/sqrt(n), 1/sqrt(n)].

	Arguments:
		matrix_shape - type = tuple: A tuple with 2 numbers, specifying the number of rows and the number of columns.

	Returns:
		output - type = Numpy array: A uniform matrix.
		
	"""

	boundaries = 1/np.sqrt(matrix_shape[1])
	return np.random.uniform(-boundaries, boundaries, matrix_shape)

# ------------------------------------------------------------------------------------------------------------------------
def xavier(matrix_shape):
	"""
	Creates a gaussian distribution matrix with a mean of 0 and variance of sqrt(2/(n+m)) where n and m are the sizes of the matrix.

	Arguments:
		matrix_shape - type = tuple: A tuple with 2 numbers, specifying the number of rows and the number of columns.

	Returns:
		output - type = Numpy array: A gaussian distribution matrix.
		
	"""

	return np.random.normal(0, np.sqrt(2/(matrix_shape[0]+matrix_shape[1])), matrix_shape)

# ------------------------------------------------------------------------------------------------------------------------
def xavier_uniform(matrix_shape):
	"""
	Creates a uniform matrix based of the xavier initializer with boundaries of [-sqrt(6)/sqrt(n+m), sqrt(6)/sqrt(n+m)] where n and m are the sizes of the matrix.

	Arguments:
		matrix_shape - type = tuple: A tuple with 2 numbers, specifying the number of rows and the number of columns.

	Returns:
		output - type = Numpy array: A uniform matrix.
		
	"""

	boundaries = np.sqrt(6)/np.sqrt(matrix_shape[0]+matrix_shape[1])
	return np.random.uniform(-boundaries, boundaries, matrix_shape)

# ------------------------------------------------------------------------------------------------------------------------
def sigmoid_uniform(matrix_shape):
	"""
	Creates a uniform matrix based of the xavier_uniform initializer with boundaries of [-(4*sqrt(6))/sqrt(n+m), 4*sqrt(6)/sqrt(n+m)] where n and m are the sizes of the matrix.

	Arguments:
		matrix_shape - type = tuple: A tuple with 2 numbers, specifying the number of rows and the number of columns.

	Returns:
		output - type = Numpy array: A uniform matrix.
		
	"""

	boundaries = 4*np.sqrt(6)/np.sqrt(matrix_shape[0]+matrix_shape[1])
	return np.random.uniform(-boundaries, boundaries, matrix_shape)

# ------------------------------------------------------------------------------------------------------------------------
def relu(matrix_shape):
	"""
	Creates a gaussian distribution matrix with a mean of 0 and variance of sqrt(2/m) where m is the number of rows

	Arguments:
		matrix_shape - type = tuple: A tuple with 2 numbers, specifying the number of rows and the number of columns.

	Returns:
		output - type = Numpy array: A gaussian distribution matrix.
		
	"""

	return np.random.normal(0, np.sqrt(2/matrix_shape[0]), matrix_shape)

# ------------------------------------------------------------------------------------------------------------------------
def relu_uniform(matrix_shape):
	"""
	Creates a uniform matrix based of the relu initializer with boundaries of [-sqrt(6/m), sqrt(6/m)] where m is the number of rows

	Arguments:
		matrix_shape - type = tuple: A tuple with 2 numbers, specifying the number of rows and the number of columns.

	Returns:
		output - type = Numpy array: A uniform matrix.
		
	"""

	boundaries = np.sqrt(6/matrix_shape[0])
	return np.random.uniform(-boundaries, boundaries, matrix_shape)
