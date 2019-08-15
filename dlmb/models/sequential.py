import json
import numpy as np
import layers import dense, batchnorm, dropout
import losses as L
import activations as A
import optimizers as O

class Sequential:
	def __init__(self, model=[], *args, **kwargs):
		"""
		The Sequential class is a linear stack of layers that allows easy communication between different layers.

		Arguments:
			model - list: A list of layers.

		"""

		# Checks if the user has sent through anything other than a list.
		if type(model) != list:
			raise ValueError("At Sequential(): Expected 'list' for the argument model, recieved '%s'" % type(model))

		self.model = model

	# ------------------------------------------------------------------------------------------------------------------------
	def add(self, layer, index=-1, *args, **kwargs):
		"""
		Adds a layer to the model at a specific index.

		Arguments:
			layer - instance of a layer class: A layer class such as 'Dense' or 'Conv'.
			index - int: An index at which to place the layer in self.model.
	
		"""

		if len(self.model) == 0:
			self.model.append(layer)
		else:
			if index == -1: # For some reason list.insert() doesn't like -1 as the index value so i've got to do it like this.
				self.model.append(layer)
			else:
				self.model.insert(index, layer)

	# ------------------------------------------------------------------------------------------------------------------------
	def delete(self, index=-1, *args, **kwargs):
		"""
		Deletes (pops) a layer from self.model at a specific index.

		Arguments:
			index - int: An index at which to delete the layer in self.model.

		"""

		self.model.pop(index)

	# ------------------------------------------------------------------------------------------------------------------------
	def build(self, loss_function="mean_squared_error", optimizer="GD", *args, **kwargs):
		"""
		Goes through each layer calling the layers' set_vars() function and sets the model's loss_function and optimizer.

		Arguments:
			loss_function - instance of a class: A mathematical function to determine how wrong the model's prediction on a data sample is. 
			optimizer - instance of a class: A class that takes each layer's gradients and optimizes them to reach a local optima in the error faster.

		"""

		for layer in self.model:
			layer.set_vars()

		# Sets the model's loss_function and optimizer.
		self.loss_function = L.get(loss_function)
		self.optimizer = O.get(optimizer)

	# ------------------------------------------------------------------------------------------------------------------------
	def get_summary(self, index="all", show_info=True, with_vars=False, *args, **kwargs):
		"""
		get_summary() returns a summary of all the layers' or a specific layer's features.

		Arguments:
			index - str/int: The index of the layer the user wants to see.
			show_info - bool: If True, get_summary() will print the layer's summary to the terminal.
			with_vars - bool: If True, get_summary() includes the layer's variables' values in the summary.

		Returns:
			summary - list: A list of dicts of the layers features.

		"""

		summaries = []

		if type(index) == int:
			if show_info:
				print("layer %s:" % str(index))
				summary = self.model[index].get_summary(with_vars)
				for key in summary:
					print("    %s: %s" % (str(key), str(summary[key])))
				print("")

			summaries.append(self.model[index].get_summary(with_vars))

		else:
			for i in range(len(self.model)):
				if show_info:
					print("layer %s:" % str(i))
					summary = self.model[i].get_summary(with_vars)
					for key in summary:
						print("    %s: %s" % (str(key), str(summary[key])))
					print("")

				summaries.append(self.model[i].get_summary(with_vars))

		return summaries

	# ------------------------------------------------------------------------------------------------------------------------
	def save(self, file_path="./saved_dlmb_model.json", *args, **kwargs):
		"""
		Using the JSON package, the model and all it's layers' and corresponding vars get saved into a .json file.

		Arguments:
			file_path - str: A file path at which the .json file gets saved.
	
		"""
		
		# First, gather all the needed data for each layer into a dict.
		saved_model = {}

		for i in range(len(self.model)):
			saved_model[f"layer{i}"] = {}
			summary = self.get_summary(index=i, show_info=False, with_vars=True)[0]
			for key in summary:
				saved_model["layer%s" % i][key] = summary[key]

		# Open or create a text file at file_path and dump the json data into there.
		with open(file_path, "w+") as json_file:
			json_file.write(json.dumps(saved_model, indent=2))

	# ------------------------------------------------------------------------------------------------------------------------
	def load(self, file_path="./saved_dlmb_model.json", *args, **kwargs):
		"""
		Goes to the file where the model has been saved and retrieves the data.

		Arguments:
			file_path - str: A file path at which the .json file has been saved.

		"""

		# Try to open the file at file_path.
		try:
			with open(file_path, "r") as json_file:
				model_layers = json.loads(json_file.read())
				index = 0

				while True:
					try:
						layer_data = model_layers[f"layer{index}"]
						if layer_data["name"] == "Dense":
							self.model.append(dense.Dense(1))

						elif layer_data["name"] == "Batchnorm":
							self.model.append(batchnorm.Batchnorm())

						elif layer_data["name"] == "Dropout":
							self.model.append(dropout.Dropout())

						self.model[-1].load(layer_data)

					# Gets called if the while loop can't find any more layers in the file.
					except Exception as e:
						break

					index += 1

		# Gets called if the program can't find the file_path.
		except Exception as e:
			raise FileNotFoundError("Can't find file path %s. Try saving the model or enter a correct file path." % file_path)

	# ------------------------------------------------------------------------------------------------------------------------
	def get_batches(self, data, batch_size, *args, **kwargs):
		"""
		Splits the data into batches with size batch_size.

		Arguments:
			data - numpy array: A numpy array of data.
			batch_size - int: A number determining the size of the batches.

		Returns:
			batches - list: A list of matrices containing the batches of the split x_features and y_features.

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
	def predict(self, data, *args, **kwargs):
		""" 
		Goes through each layer feeding forward the output from the last layer to the next with the layer's map_data() function. The starting output is the data var.

		Arguments:
			data - numpy array: A numpy array containing real numbers that are labels for real world data.

		Returns:
			data - numpy array: A numpy array with size (batch size, final layer's output size).

		"""

		for layer in self.model:
			data = layer.map_data(data)
		return data

	# ------------------------------------------------------------------------------------------------------------------------
	def train(self, x_features, y_features, epochs=1, batch_size=1, shuffle=True, show_epoch_info=True, show_batch_info=False, *args, **kwargs):
		"""
		Gets the model's current state by calling the predict() function and passing through x_features as an argument.
		The train() function then gets the model's error with self.loss_function.
		Then the gradients for each var for each layer are computed and then the vars are updated with those gradients.

		Arguments:
			x_features - numpy array: A matrix or vector (depending on batch size) containing real numbers that are labels for real world data.
			y_features - numpy array: A matrix or vector (depending on batch size) containing real numbers that are the correct labels for the output of the model.
			epochs - int: A number that is the amount of times the model trains on x_features and y_features.
			batch_size - int: A number determining the size of the batches that x_features and y_features will be split into.
			shuffle - bool: if true, shuffles the x_features and y_features but makes sure they still relate to each other.
			show_epoch_info - bool: if true, prints to the terminal the average loss and accuracy for each epoch.
			show_batch_info - bool: if true, prints to the terminal the average loss and accuracy for each batch in a single epoch.

		Returns:
			cached_error - list: A list containing the total error for each batch in each epoch.

		"""

		# Shuffle if shuffle=True.
		if shuffle:
			assert len(x_features) == len(y_features)
			p = np.random.permutation(len(x_features))
			x_features = x_features[p]
			y_features = y_features[p]

		# Stores the cached_error.
		cached_error = []

		# Goes through each epoch.
		for epoch in range(epochs):

			# Stores the epoch_error.
			epoch_error = []

			# Turn the data into batches.
			x_batches = self.get_batches(x_features, batch_size)
			y_batches = self.get_batches(y_features, batch_size)

			# Goes through each batch.
			for i in range(len(x_batches)):
				X = x_batches[i]
				Y = y_batches[i]

				# Get the model's current state and the batch error and accuracy.
				current_state = self.predict(X)
				batch_error = np.mean(self.loss_function.map_data(current_state, Y))

				# Calculate the gradients for each var in each layer.
				layer_error = self.loss_function.calculate_gradients(current_state, Y)

				for layer in reversed(self.model):
					layer_error = layer.calculate_gradients(layer_error)

				# Update the vars in each layer with the calculated gradients.
				for layer in self.model:
					layer.update_vars(self.optimizer, epoch+1)

				cached_error.append(batch_error)
				epoch_error.append(batch_error)

				# Prints to the terminal the average loss and accuracy for each batch if show_batch_info=True.			
				if show_batch_info:	
					print("Batch: %s; Loss: %s;" % (i+1, np.mean(np.asarray(batch_error))))


			# Prints to the terminal the average loss and accuracy for each epoch if show_epoch_info=True.		
			if show_epoch_info:	
				print("Epoch: %s; Loss: %s;" % (epoch+1, np.mean(np.asarray(epoch_error))))

		return cached_error
