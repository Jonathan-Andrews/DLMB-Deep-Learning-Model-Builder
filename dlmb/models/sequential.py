import numpy as np
import json
import layers as LA
import losses as L
import activations as A
import optimizers as O

class Sequential:
	def __init__(self, model=[]):

		"""
		The Sequential model class handles an array of neural-net layers by allowing the layers to interact with each other.
		It allows the user to call some commands and not have to write extra code so all the layers can "talk" to each other.

		Arguments:
			model - type = Array: An array of layers that the user wishes to use.

		"""

		self.model = model

	# ------------------------------------------------------------------------------------------------------------------------
	def add(self, layer):
		"""
		add_model() allows the user to add a new layer to the self.model array if it wasn't specified before.

		Arguments:
			layer - type = Instance of a class: A neural-net layer from the layers folder.

		"""

		self.model.append(layer)

	# ------------------------------------------------------------------------------------------------------------------------
	def build(self, loss_function=L.Mean_squared_error(), optimizer=O.GD()):
		"""
		Goes through each layer calling the layers set_vars() function and sets the model's loss_function and optimizer.

		Arguments:
			loss_function - type = Instance of a class: A mathematical function to determine how wrong the model's prediction on a data sample is. 
			optimizer - type = Instance of a class: A class that takes each layer's gradients and optimizes them to reach a local optima in the error faster.

		"""

		for layer in self.model:
			layer.set_vars()

		# Sets the model's loss_function if the user sent through a string as the argument.
		self.loss_function = loss_function

		if type(self.loss_function) == str:
			if self.loss_function.lower() in ("mse", "mean_squared_error"):
				self.loss_function = L.Mean_squared_error()
			elif self.loss_function.lower() in ("bce", "binary_crossentropy"):
				self.loss_function = L.Binary_crossentropy()
			elif self.loss_function.lower() in ("ce", "crossentropy"):
				self.loss_function = L.Crossentropy()
			else:
				raise ValueError(f"{self.loss_function} is not currently an available loss function.")

		# Sets the model's optimizer if the user sent through a string as the argument.
		self.optimizer = optimizer

		if type(self.optimizer) == str:
			if self.optimizer.lower() in ("gd", "gradient_decent"):
				self.optimizer = O.GD()
			elif self.optimizer.lower() in ("rmsprop"):
				self.optimizer = O.RMSprop()
			elif self.optimizer.lower() in ("adam"):
				self.optimizer = O.Adam()
			else:
				raise ValueError(f"{self.optimizer} is not currently an available optimizer.")

	# ------------------------------------------------------------------------------------------------------------------------
	def save(self, file_path="./saved_dlmb_model.json"):
		"""
		Using the JSON package, the neural-net model and all it's layers and corresponding vars get saved into a .json file.

		Arguments:
			file_path - type = String: A file path at which the .json file gets saved.
	
		"""
		
		# First, gather all the needed data for each layer into a dict.
		saved_model = {}

		for i in range(len(self.model)):
			if self.model[i].layer_type == "Dense":
				saved_model[f"layer{i}"] = {
												"layer_type":"Dense",
												"built":self.model[i].built,
												"layer_shape":self.model[i].layer_shape,
												"activation":self.model[i].activation.name,
												"bias_type":self.model[i].bias_type,
												"weight_initializer":self.model[i].weight_initializer,
												"bias_initializer":self.model[i].bias_initializer,
												"weight_regularizer":self.model[i].weight_regularizer.name,
												"bias_regularizer":self.model[i].bias_regularizer.name,
												"weight_optimizations":np.asarray(self.model[i].weight_optimizations)[0:].tolist(),
												"bias_optimizations":np.asarray(self.model[i].bias_optimizations)[0:].tolist(),
												"weights":self.model[i].weights.tolist(),
												"bias":self.model[i].bias.tolist()
										   }

			elif self.model[i].layer_type == "Batchnorm":
				saved_model[f"layer{i}"] = {
												"layer_type":"Batchnorm",
												"built":self.model[i].built,
												"beta_initializer":self.model[i].beta_initializer,
												"gamma_initializer":self.model[i].gamma_initializer,
												"beta_regularizer":self.model[i].beta_regularizer.name,
												"gamma_regularizer":self.model[i].gamma_regularizer.name,
												"epsilon":self.model[i].epsilon,
												"beta_optimizations":np.asarray(self.model[i].beta_optimizations)[0:].tolist(),
												"gamma_optimizations":np.asarray(self.model[i].gamma_optimizations)[0:].tolist(),
												"beta":self.model[i].beta.tolist(),
												"gamma":self.model[i].gamma.tolist()
										   }

			elif self.model[i].layer_type == "Dropout":
				saved_model[f"layer{i}"] = {
												"layer_type":"Dropout",
												"keep_prob":self.model[i].keep_prob
										   }

		# Open or create a text file at file_path and dump the json data into there.
		with open(file_path, "w+") as json_file:
			json_file.write(json.dumps(saved_model, indent=2))

	# ------------------------------------------------------------------------------------------------------------------------
	def load(self, file_path="./save_dlmb_model.json"):
		"""
		Goes to the file where the model has been saved and retrieves the data.

		Arguments:
			file_path - type = String: A file path at which the .json file has been saved.

		"""

		# Try to open the file at file_path
		try:
			with open(file_path, "r") as json_file:
				model_layers = json.loads(json_file.read())
				index = 0

				while True:
					try:
						layer_data = model_layers[f"layer{index}"]

						if layer_data["layer_type"] == "Dense":
							self.model.append(LA.dense.Dense(layer_data["layer_shape"]))
							self.model[-1].built = layer_data["built"]
							self.model[-1].activation = layer_data["activation"]
							self.model[-1].bias_type = layer_data["bias_type"]
							self.model[-1].weight_initializer = layer_data["weight_initializer"]
							self.model[-1].bias_initializer = layer_data["bias_initializer"]
							self.model[-1].weight_regularizer = layer_data["weight_regularizer"]
							self.model[-1].bias_regularizer = layer_data["bias_regularizer"]
							self.model[-1].weight_optimizations = np.asarray(layer_data["weight_optimizations"])
							self.model[-1].bias_optimizations = np.asarray(layer_data["bias_optimizations"])
							self.model[-1].weights = np.asarray(layer_data["weights"])
							self.model[-1].bias = np.asarray(layer_data["bias"])

						elif layer_data["layer_type"] == "Batchnorm":
							self.model.append(LA.batchnorm.Batchnorm())
							self.model[-1].built = layer_data["built"]
							self.model[-1].beta_initializer = layer_data["beta_initializer"]
							self.model[-1].gamma_initializer = layer_data["gamma_initializer"]
							self.model[-1].beta_regularizer = layer_data["beta_regularizer"]
							self.model[-1].gamma_regularizer = layer_data["gamma_regularizer"]
							self.model[-1].epsilon = layer_data["epsilon"]
							self.model[-1].beta_optimizations = np.asarray(layer_data["beta_optimizations"])
							self.model[-1].gamma_optimizations = np.asarray(layer_data["gamma_optimizations"])
							self.model[-1].beta = np.asarray(layer_data["beta"])
							self.model[-1].gamma = np.asarray(layer_data["gamma"])

						elif layer_data["layer_type"] == "Dropout":
							self.model.append(LA.dropout.Dropout())
							self.model[-1].keep_prob = layer_data["keep_prob"]

					# Gets called if the while loop can't find any more layers in the file
					except Exception as e:
						break

					index += 1

		# Gets called if the program can't find the file_path
		except Exception as e:
			raise FileNotFoundError(f"Cant find file path {file_path}. Try saving the model or enter a correct file path.")

	# ------------------------------------------------------------------------------------------------------------------------
	def get_batches(self, data, batch_size):
		"""
		Splits the data into batches with size batch_size.

		Arguments:
			data - type = Numpy array: A concatenated matrix of the train() function's x_features and y_features.
			batch_size - type = Int: A number determining the size of the batches.

		Returns:
			batches - type = Array: An array of matrices containing the batches of the split x_features and y_features.

		"""

		# Stores the batches.
		batches = []

		index = 0
		for _ in range(len(data)):
			# If the batch_size is bigger than the remaining data then just put the remaining data into a single batch and then break.
			if batch_size > len(data[index:]):
				batches.append(data[index:])
				break
			else:
				batches.append(data[index:index+batch_size])

			index += batch_size

		# Fixes a problem that occurs when batch_size is bigger than the remaining data.
		if len(batches[-1]) == 0:
			batches = np.delete(batches, -1)

		return batches

	# ------------------------------------------------------------------------------------------------------------------------
	def predict(self, data):
		""" 
		Goes through each layer feeding forward the output from the last layer to the next with the layer's map_data() function. The starting output is the data var.

		Arguments:
			data - type = Numpy array: A matrix or vector (depending on batch size) containing real numbers that are labels for real world data.

		Returns:
			data - type = Numpy array: A matrix with size (batch size, final layer's output size).

		"""

		for layer in self.model:
			data = layer.map_data(data)
		return data

	# ------------------------------------------------------------------------------------------------------------------------
	def train(self, x_features, y_features, epochs=1, batch_size=1, shuffle=True, show_epoch_info=False, show_batch_info=False):
		"""
		Gets the model's current state by calling the predict() function and passing through x_features as an argument.
		The train() function then gets the model's error with self.loss_function.
		Then the gradients for each var for each layer are computed and then the vars are updated with those gradients.

		Arguments:
			x_features - type = Numpy array: A matrix or vector (depending on batch size) containing real numbers that are labels for real world data.
			y_features - type = Numpy array: A matrix or vector (depending on batch size) containing real numbers that are the correct labels for the output of the model.
			epochs - type = Int: A number that is the amount of times the model trains on x_features and y_features.
			batch_size - type = Int: A number determining the size of the batches that x_features and y_features will be split into.
			shuffle - type = Bool: if true, shuffles the x_features and y_features but makes sure they still relate to each other.
			show_epoch_info - type = Bool: if true, prints to the terminal the average loss and accuracy for each epoch.
			show_batch_info - type = Bool: if true, prints to the terminal the average loss and accuracy for each batch in a single epoch.

		Returns:
			total_error - type = Array: An array containing the total error for each batch in each epoch.

		"""

		# Concatenate the x_features and y_features together to split into batches and to shuffle them if shuffle=True.
		data = np.concatenate((x_features, y_features), axis=1)

		if shuffle:
			np.random.shuffle(data)

		# Stores the total_error.
		total_error = []

		# Goes through each epoch.
		for epoch in range(epochs):

			# Stores the batch_errors.
			batch_errors = []

			# Turn the data into batches.
			batches = self.get_batches(data, batch_size)

			# Goes through each batch.
			for i in range(len(batches)):
				X = batches[i][0:, 0:batches[i].shape[1]-y_features.shape[1]]
				Y = batches[i][0:, x_features.shape[1]:]

				# Get the model's current state and the batch error and accuracy.
				current_state = self.predict(X)
				batch_error = np.mean(self.loss_function.function(current_state, Y))

				# Calculate the gradients for each var in each layer.
				layer_error = self.loss_function.derivative(current_state, Y)

				for layer in reversed(self.model):
					layer_error = layer.calculate_gradients(layer_error)

				# Update the vars in each layer with the calculated gradients.
				for layer in self.model:
					layer.update_vars(self.optimizer, epoch+1)

				total_error.append(batch_error)
				batch_errors.append(batch_error)

				# Prints to the terminal the average loss and accuracy for each epoch if show_info=True			
				if show_batch_info:	
					print(f"Batch: {i+1}; Loss: {np.mean(np.asarray(batch_error))};")


			# Prints to the terminal the average loss and accuracy for each epoch if show_info=True			
			if show_epoch_info:	
				print(f"Epoch: {epoch+1}; Loss: {np.mean(np.asarray(batch_errors))};")

		return total_error