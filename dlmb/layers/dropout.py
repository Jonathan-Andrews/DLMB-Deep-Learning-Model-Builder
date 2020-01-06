from layers.base_layer import *
from utils.function_helpers import *


class Dropout(Base_Layer):
	@accepts(self="any", keep_prob=float)
	def __init__(self, keep_prob=0.5) -> None:
		"""
			Dropout is a type of regularization that creates a mask or probabilities.
			This mask will then be applied to any incoming input, 
			in effect cancelling a certain percentage of the input.

			Arguments:
				keep_prob : float : The probability that each feature along each row of the input will be kept.

		"""

		super().__init__("Dropout")

		self.keep_prob = keep_prob
		self.built = False


	def build(self) -> None:
		self.built = True


	@accepts(self="any", with_vars=bool)
	def get_summary(self, with_vars) -> dict:
		"""
			get_summary() returns a summary of the layers features.

			Arguments:
				with_vars : bool : If True, get_summary() includes the layer's variables' values in the summary.

			Returns:
				summary : dict : A dictonary of the layers features.

		"""

		summary = {
					"name":self.name,
					"built":self.built,
					"keep_prob":self.keep_prob
				  }

		return summary


	@accepts(self="any", layer_data=dict)
	def load(self, layer_data) -> None:
		"""
			Takes the layer_data from the model this layer belongs to, and sets all vars equal to each key in layer_data.

			Arguments:
				layer_data : dict : A dictonary of saved vars from when this layer was first built and then saved.

		"""

		self.name = layer_data["name"]
		self.built = layer_data["built"]
		self.keep_prob = layer_data["keep_prob"]


	@accepts(self="any", data=np.ndarray)
	def map_data(self, data) -> np.ndarray:
		"""
			Applies a mask to some data in the form of (x * ([...] < keep_prop))/keep_prob.

			Arguments:
				data : np.ndarray : An n dimensional numpy array containing real numbers passed from the previous layer.

			Returns:
				output : np.ndarray : An n dimensional numpy array of outputs from this layer.

		"""

		data_shape = data.shape

		# Try to write a decorator for this.
		# Makes sure that the data is a 2d np.ndarray.
		if len(data.shape) == 1:
			data = data.reshape((1, data.shape[0]))
		elif len(data.shape) > 1:
			length = 1
			for i in range(len(data.shape)-1):
				length *= data.shape[i+1]
			data = data.reshape((data.shape[0], length))

		self.mask = np.random.random(data.shape) < self.keep_prob
		return np.reshape((data*self.mask)/self.keep_prob, data_shape)


	@accepts(self="any", error=np.ndarray)
	def calculate_gradients(self, error) -> np.ndarray:
		"""
			Calculates the gradients of the error W.R.T to the input of this layer.

			Arguments:
				error : np.ndarray : An n dimensional numpy array containing the errors for the previous layer.

			Returns:
				output : np.ndarray : An n dimensional numpy array containing the errors for this layer.

		"""

		error_shape = error.shape

		# Makes sure that the error is a 2d np.ndarray.
		if len(error.shape) == 1:
			error = error.reshape((1, error.shape[0]))
		elif len(error.shape) > 1:
			length = 1
			for i in range(len(error.shape)-1):
				length *= error.shape[i+1]
			error = error.reshape((error.shape[0], length))


		return np.reshape((error*self.mask)/self.keep_prob, error_shape)


	@accepts(self="any", optimizer=op.Base_Optimizer, epoch=int)
	def update_vars(self, optimizer, epoch) -> None:
		pass
