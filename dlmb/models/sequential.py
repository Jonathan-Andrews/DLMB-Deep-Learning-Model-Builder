import json
import numpy as np
import copy

from utils import *

import layers as la
import losses as lo
import optimizers as op

class Sequential:
	@accepts(self="any", model=list)
	def __init__(self, model=[]) -> None:
		"""
			The Sequential class is a linear stack of neural-net layers that allows easy communication between the different layers.

			Arguments:
				model : list[la.Base_Layer] : A list of neural-net layers that inherit from Base_Layer.

		"""

		self.model = model


	@accepts(self="any", layer=la.Base_Layer, index=int)
	def add(self, layer, index=-1) -> None:
		"""
			add() inserts a neural-net layer that inherits from Base_Layer into self.model at a specific index.

			Arguments:
				layer : la.Base_Layer : An instance of a neural-net class that inherits from Base_Layer.
				index : int   		  : The index at which to insert layer into self.model.

		"""

		if index == -1:
			self.model.insert(len(self.model), layer)  # insert() with an index of -1 doesn't actually insert to the end of the list.
		else:
			self.model.insert(index, layer)



	@accepts(self="any", index=(int, str))	
	def remove(self, index=-1) -> None:
		"""
			remove() deltes/pops an entry from self.model at a specific index.

			Arguments:
				index : int/str : An index at which to delete the entry in self.model.

		"""

		if index == "all":
			self.model = []

		elif isinstance(index, int):
			self.model.pop(index)


	@accepts(self="any", loss=(lo.Base_Loss, str), optimizer=(op.Base_Optimizer, str))
	def build(self, loss='mse', optimizer='gd') -> None:
		"""
			build() itterates through self.model, initializing all the layers' variables.

			Arguments:
				loss      : lo.Base_Loss/str      : The loss function this model will use to calculate how well the model is doing.
				optimizer : op.Base_Optimizer/str : The optimizer function this model will use to optimize the vars of each layer.

		"""

		self.loss = lo.get(loss)
		self.optimizer = op.get(optimizer)

		for layer in self.model:
			layer.build()


	@accepts(self="any", index=(str, int), show_info=bool, with_vars=bool)
	def get_summary(self, index="all", show_info=True, with_vars=False) -> list:
		"""
			
			get_summary() returns a summary of all the layers' or a specific layer's features.

			Arguments:
				index     : str/int : The index of the layer the user wants a summary of.
				show_info : bool    : If True, get_summary() will print the layer's summary to the terminal.
				with_vars : bool    : If True, get_summary() includes the layer's variables' values in the summary.

			Returns:
				summary : list : A list of dicts of the layers features.
	
		"""
		

		summaries = []

		if type(index) == int:
			summary = self.model[index].get_summary(with_vars)

			if show_info:
				print("layer %s:" % str(index))
				for key in summary:
					print("    %s: %s" % (str(key), str(summary[key])))
				print("")

			summaries.append(summary)

		# If index == "all": 
		else:
			for i in range(len(self.model)):
				summaries.append(self.get_summary(i, show_info, with_vars)[0])

		return summaries


	@accepts(self="any", file_path=str)
	def save(self, file_path="./saved_dlmb_model.json"):
		"""
			Using the JSON package, the model and all it's layers' and corresponding vars get saved into a .json file.

			Arguments:
				file_path : str : A file path where the .json file gets saved.
	
		"""
		
		# First, gather all the needed data for each layer into a dict.
		saved_model = {}

		summaries = self.get_summary(show_info=False, with_vars=True)

		for i in range(len(summaries)):
			saved_model["layer: %s" % i] = {}
			for key in summaries[i]:
				saved_model["layer: %s" % i][key] = summaries[i][key]

		# Open or create a text file at file_path and dump the json data into there.
		with open(file_path, "w+") as json_file:
			json_file.write(json.dumps(saved_model, indent=2))


	@accepts(self="any", file_path=str)
	def load(self, file_path="./saved_dlmb_model.json"):
		"""
			Goes to the file where the model has been saved and retrieves the data.

			Arguments:
				file_path : str : A file path where the .json file has been saved.

		"""


		layers = {
					"Dense":la.Dense(0, 0),
					"Batchnorm":la.Batchnorm(0),
					"Dropout":la.Dropout()
				 }

		# Try to open the file at file_path.
		try:
			with open(file_path, "r") as json_file:
				model_layers = json.loads(json_file.read())

				for i in range(len(model_layers)):
					layer_data = model_layers["layer: %s" % i]

					new_layer = copy.copy(layers[layer_data["name"]])
					new_layer.load(layer_data)
					self.model.append(new_layer)

		# Gets called if the program can't find the file_path.
		except Exception as e:
			raise FileNotFoundError("Can't find file path %s. Try saving the model or enter a correct file path." % file_path)

	
	@accepts(self="any", data=np.ndarray)
	def predict(self, data:np.ndarray) -> np.ndarray:
		""" 
			predict() feeds the output from the previous layer to the next.

			Arguments:
				data : ND_ARRAY : An n dimensional numpy array containing real numbers that are real world data.

			Returns:
				data : ND_ARRAY : An n dimensional numpy array of data that has been fed forward through each layer.

		"""

		for layer in self.model:
			data = layer.map_data(data)
		return data


	@accepts(self="any", x_features=np.ndarray, y_features=np.ndarray, epochs=int, batch_size=int,
			 shuffle=bool, epoch_info=bool, batch_info=bool)
	def train(self, x_features, y_features, epochs=1, batch_size=1, 
					shuffle=True, epoch_info=True, batch_info=False) -> list:
		"""
			train() gets the model's current state by making a prediction on x_features and comparing it y_features with self.loss.
			Then an overall loss is calculated for the model and passed backwards through each layer where all the vars will be updated.

			Arguments:
				x_features : np.ndarray  : An n dimensional numpy array containing real numbers that are real world data.
				y_features : np.ndarray  : An n dimensional numpy array containing real numbers that are the labels for x_features.
				epochs     : int       	 : The amount of time this model will train on all of x_features and y_features.
				batch_size : int         : The number of x_features and y_features in each batch of each epoch.
				shuffle    : bool        : If true, x_features and y_features will be shuffles but they will still relate to each other.
				epoch_info : bool        : If true, prints to the terminal the average loss and accuracy for each epoch.
				batch_info : bool        : If true, prints to the terminal the average loss and accuracy for each batch in a single epoch.

			Returns:
				list:
					cached_error    : np.ndarray : An n dimensional numpy array containing the total error for each batch in each epoch.
					cached_accuracy : np.ndarray : An n dimensional numpy array containing the total accuracy for each batch in each epoch.

		"""

		if len(x_features.shape) > 2:
			x_shape = x_features.shape[1:]
			x_features = x_features.reshape(*x_features.shape[:1], -1)

		# Stores the cached_error.
		cached_error = []
		cached_accuracy = []

		# Goes through each epoch.
		for epoch in range(epochs):

			# Stores the epoch_error.
			epoch_error = []
			epoch_accuracy = []


			# Shuffle if shuffle=True.
			if shuffle:
				assert len(x_features) == len(y_features)
				p = np.random.permutation(len(x_features))
				x_features = x_features[p]
				y_features = y_features[p]

			# Turn the data into batches.

			x_batches = [np.squeeze(segment, axis=0) for segment, _, _, _, _ in get_segments(x_features, batch_size, x_features.shape[1], (batch_size, 0))]
			y_batches = [np.squeeze(segment, axis=0) for segment, _, _, _, _ in get_segments(y_features, batch_size, y_features.shape[1], (batch_size, 0))]


			# Goes through each batch.
			for i in range(len(x_batches)):

				# Get the individual batches.
				x = x_batches[i].reshape((batch_size, ) + x_shape)
				y = y_batches[i]


				# Get the model's current state and the batch error and accuracy.
				current_state = self.predict(x)

				# Store the data
				batch_error = np.mean(self.loss.map_data(y, current_state))
				batch_accuracy = 0 # Work on this later.


				# Calculate the gradients for each var in each layer.
				layer_error = self.loss.calculate_gradients(y, current_state)

				# Calculate the gradients for each layer.
				for layer in reversed(self.model):
					layer_error = layer.calculate_gradients(layer_error)

				# Update the vars in each layer with the calculated gradients.
				for layer in self.model:
					layer.update_vars(self.optimizer, epoch+1)


				# Store all the info for the batch.
				cached_error.append(batch_error)
				cached_accuracy.append(batch_accuracy)

				epoch_error.append(batch_error)
				epoch_accuracy.append(batch_accuracy)


				# Prints to the terminal the average loss and accuracy for each batch if batch_info=True.			
				if batch_info:	
					print("Batch: %s; Loss: %s; Accuracy: %s;" % (i+1, batch_error, batch_accuracy))

			# Prints to the terminal the average loss and accuracy for each epoch if epoch_info=True.		
			if epoch_info:	
				print("Epoch: %s; Loss: %s; Accuracy: %s;" % (epoch+1, np.mean(np.asarray(epoch_error)), np.mean(np.asarray(epoch_accuracy))))

		return cached_error, cached_accuracy
