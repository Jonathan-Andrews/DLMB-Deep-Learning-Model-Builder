from layers.base_layer import *
from utils import *

import activations as a
import initializers as i


class Conv(Base_Layer):
	@accepts(self="any", output_size=int, input_shape=tuple, activation=(a.Base_Activation, str),
			  bias_type=str, trainable=bool, filter_size=(tuple,int), stride=(tuple,int), padding=int,
			  weight_initializer=str, bias_initializer=str)

	def __init__(self, output_size, input_shape, activation="relu", bias_type="per_node", 
				 trainable=True, filter_size=(1,1), stride=(1,1), padding=0,
				 weight_initializer="random", bias_initializer="random") -> None:

		"""
			The conv layer or convolutional layer creates a numpy array that is convolved or cross-correlated over some data.
			The conv layer also makes use of shared variables, meaning that compared to a dense layer there will be less variables.

			Arguments:
				output_size        : int                 : An int of the output size, (E.X output_size=6 returns a numpy array of size 
				                                         - [batch_size, ..., 6] "an image with 6 channels").
				input_shape        : tuple               : A tuple of the input shape, (E.X input_shape=(28, 28, 1) which in this example is a 28*28 image with 1 channel).
				activation         : Base_Activation/str : A mathematical function that generally applies a non-linear mapping to some data.
				bias_type          : str      	         : Has three settings, per_node (a bias with weights for every output), 
												         - single (one bias weight for all outputs), none (no bias is used).
				trainable          : bool     	         : If True, the vars in this layer will update based on the calculated loss of this layer W.R.T the vars.
				filter_size        : tuple/int           : A tuple of 2 values or a int specifying the height and width of each segment of data that will be convolved over.
				stride             : tuple/int           : A tuple or int of values specifying the stride for the height and width when convolving over some data.
				padding            : int                 : An int specifying the amount of padding to be added around the input data.
				weight_initializer : str                 : An initializer function to set the values for the weights.
				bias_initializer   : st                  : An initializer function to set the values for the bias'.

		"""

		super().__init__("Conv")

		if len(input_shape) == 2:
			if channels_first:
				input_shape = (1, input_shape[0], input_shape[1])
			else:
				input_shape = (input_shape[0], input_shape[1], 1)

		if isinstance(filter_size, int):
			filter_size = (filter_size, filter_size)

		if isinstance(stride, int):
			stride = (stride, stride)


		self.built = False
		self.layer_shape = (output_size, input_shape)
		self.activation = a.get(activation)
		self.bias_type = bias_type
		self.trainable = trainable

		self.filter_size = filter_size
		self.stride = stride
		self.padding = padding

		self.weight_initializer = weight_initializer
		self.bias_initializer = bias_initializer

		self.weights = 0
		self.bias = 0

		self.optimizations = {"weight":[0,0], "bias":[0,0]}
		self.cached_data = {}

		# Put this into it's own function called get_ouput_size()
		# calculates the output sizes for the layer.
		height_output_size = int((self.layer_shape[1][0]-self.filter_size[0] + (2*self.padding))/self.stride[0]) + 1
		width_output_size = int((self.layer_shape[1][1]-self.filter_size[1] + (2*self.padding))/self.stride[1]) + 1
		self.output_shape = (height_output_size, width_output_size, output_size)


	def build(self) -> None:
		"""
			Sets the values for all of the vars that will be used.
			Also sets the layers activation function if the user sent through a string as the argument.

		"""

		if not self.built:
			
			# Set self.weights.
			init_shape = (self.output_shape[2], self.filter_size[0]*self.filter_size[1]*self.layer_shape[1][2])
			weight_shape = (self.output_shape[2], self.filter_size[0]*self.filter_size[1], self.layer_shape[1][2])
			self.weights = i.get(self.weight_initializer)(weight_shape).reshape(weight_shape)

			# Set self.bias.
			self.bias = i.get(self.weight_initializer)((self.layer_shape[0], 1))

			# Set self.bias based on self.bias_type.
			if self.bias_type == "per_node":
				self.bias = self.bias
			elif self.bias_type == "single":
				self.bias = self.bias[0]
			elif self.bias_type == "none":
				self.bias == np.array([[0]])
			else:
				print("At Conv.set_vars(): '%s' is not a valid bias_type. Has been set to 'per_node' by default" % self.bias_type)

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
					"layer_shape":self.layer_shape,
					"activation":self.activation.name,
					"bias_type":self.bias_type,
					"trainable":self.trainable,
					"filter_size":self.filter_size,
					"stride":self.stride,
					"padding":self.padding,
					"output_size":self.output_size,
					"weight_initializer":self.weight_initializer,
					"bias_initializer":self.bias_initializer,
					"num_trainable_vars":(self.weights.shape[0]*self.weights.shape[1])+self.bias.shape[0]
				  }

		if with_vars:
			summary["weights"] = np.asarray(self.weights)
			summary["bias"] = np.asarray(self.bias)
			summary["optimizations"] = {key:np.asarray(self.optimizations[key]).tolist() for key in self.optimizations}

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
		self.layer_shape = (layer_data["layer_shape"][0], tuple(layer_data["layer_shape"][1]))
		self.activation = A.get(layer_data["activation"])
		self.bias_type = layer_data["bias_type"]
		self.trainable = layer_data["trainable"]
		self.filter_size = layer_data["filter_size"]
		self.stride = layer_data["stride"]
		self.padding = layer_data["padding"]
		self.output_size = layer_data["output_size"]
		self.weight_initializer = layer_data["weight_initializer"]
		self.bias_initializer = layer_data["bias_initializer"]
		self.weights = np.asarray(layer_data["weights"])
		self.bias = np.asarray(layer_data["bias"])
		self.optimizations = {key:np.asarray(layer_data["optimizations"][key]) for key in layer_data["optimizations"]}


	@accepts(self="any", data=np.ndarray)
	def map_data(self, data) -> np.ndarray:
		"""
			Maps the data to an output by cross-correlating weights over the data.

			Arguments:
				data : np.ndarray : An n dimensional numpy array containing real numbers passed from the previous layer.

			Returns:
				output : np.ndarray : An n dimensional numpy array of outputs from this layer.

		"""

		d_shape = data.shape

		# Turn the data into an easier to use shape.
		if len(d_shape) == 2:
			data = np.expand_dims(np.expand_dims(data, axis=0), axis=3)
		elif len(d_shape) == 4:
			data = data
		else:
			raise ValueError("At Conv.map_data(): Data with shape %s can not be used. Please reshape the data to a 2dim or 4dim sized numpy array." % d_shape)

		# Store the current data and perform mathmatical operations to it.		
		self.cached_data["data"] = data

		# Make sure the shape of the data matches this layers input_shape.
		if data.shape[1:] == self.layer_shape[1]:

			# Create a new numpy array to hold the mapped data.
			self.cached_data["mapped_data"] = np.zeros((data.shape[0], self.output_shape[0], self.output_shape[1], self.output_shape[2]))

			for segment, height, width, _, _ in get_segments(data, self.filter_size[0], self.filter_size[1], self.stride):
				segment = np.expand_dims(segment.reshape(*segment.shape[:1], -1, *segment.shape[-1:]), axis=1)
				self.cached_data["mapped_data"][0:, height, width, 0:] += np.sum(segment*self.weights, axis=(2,3))+self.bias.T

			return self.activation.map_data(self.cached_data["mapped_data"])

		else:
			raise ValueError("At Conv.map_data(): Data with shape %s does not match this layers specified input_shape of %s" % (d_shape[1:], self.layer_shape[1]))


	@accepts(self="any", error=np.ndarray)
	def calculate_gradients(self, error) -> np.ndarray:
		"""
			Calculates the gradients of the error W.R.T each trainable var in the layer.

			Arguments:
				error : np.ndarray : An n dimensional numpy array containing the errors for the previous layer.

			Returns:
				output : np.ndarray : An n dimensional numpy array containing the errors for this layer.

		"""

		error = error.reshape((error.shape[0], self.output_shape[0], self.output_shape[1], self.output_shape[2]))
		dO_dZ = self.activation.calculate_gradients(self.cached_data["mapped_data"])
		node_errors = error*dO_dZ.reshape(error.shape)


		# Calculate the weight gradients
		self.optimizations["weight"][0] = np.zeros(self.weights.shape)
		for segment, height, width, _, _ in get_segments(self.cached_data["data"], self.filter_size[0], self.filter_size[1], self.stride):

			# Sort out the shapes so they can be broadcast together.
			segment = np.expand_dims(segment, axis=1)
			error_segment = np.expand_dims(node_errors[0:, height:height+1, width:width+1], axis=4).transpose(0,3,1,2,4)

			segment = segment.reshape(*segment.shape[:2], -1, *segment.shape[-1:])
			error_segment = error_segment.reshape(*error_segment.shape[:2], -1, *error_segment.shape[-1:])

			self.optimizations["weight"][0] += np.mean(error_segment*segment, axis=0)


		# Calculate the bias gradients based on self.bias_type.		
		if self.bias_type == "per_node":
			self.optimizations["bias"][0] = np.mean(np.sum(node_errors, axis=(1,2)), axis=0, keepdims=True).T
		elif self.bias_type == "single":
			self.optimizations["bias"][0] = np.mean(np.sum(node_errors, axis=(1,2,3)))
		else:
			self.optimizations["bias"][0] = np.array([[0.0]])


		# Return this layer's error for the next layer to use.
		dE_dI = np.zeros(self.cached_data["data"].shape)

		# Sort out the shapes so they can be broadcast together.
		weights = np.expand_dims(self.weights.reshape(self.output_shape[2], self.filter_size[0], self.filter_size[1], self.layer_shape[1][2]), axis=0)
		error_segments = np.expand_dims(node_errors, axis=1).transpose(0, 4, 2, 3, 1)

		for _, height, width, i, j in get_segments(self.cached_data["data"], self.filter_size[0], self.filter_size[1], self.stride):
			error_segment = error_segments[0:, 0:, height:height+1, width:width+1, 0:]
			dE_dI[0:, i:i+self.filter_size[0], j:j+self.filter_size[1], 0:] += np.sum(error_segment*weights, axis=1)

		return dE_dI

	@accepts(self="any", optimizer=op.Base_Optimizer, epoch=int)
	def update_vars(self, optimizer, epoch) -> None:
		"""
			Updates the trainable vars in the layer with the correct gradients.

			Arguments:
				optimizer : base_optimizer : An optimizer class that takes each layer's gradients and optimizes them to reach a local optima in the error faster.
				epoch     : int       	   : The current epoch that the layer is training on.

		"""

		if self.trainable:
			self.optimizations["weight"] = optimizer.map_data(self.optimizations["weight"], epoch)
			self.optimizations["bias"] = optimizer.map_data(self.optimizations["bias"], epoch)

			self.weights -= self.optimizations["weight"][0]
			self.bias -= self.optimizations["bias"][0]