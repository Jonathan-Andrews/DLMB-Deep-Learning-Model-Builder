import numpy
from utils.function_helpers import *

@accepts(data=np.ndarray, segment_height=int, segment_width=int, stride=(tuple,int))
def get_segments(data, segment_height, segment_width, stride=(1,1)):
	"""
		Splits the data into segments based on segment height and width.

		Arguments:
			data           : np.ndarray : A 3d numpy array of data that has been passed to this layer.
			segment_height : int        : The height for each segment of data.
			segment_width  : int        : The width for each segment of data.
			stride         : tuple/int  : The amount to move across the data on each itteration.

		Yields:
			segment : np.ndarray : The next gettable segment of data.
			height  : int        : The current height index of the segment of data
			width   : int        : The current width index of the segment of data
			h_index : int        : The current height index of the data
			w_index : int        : The current width index of the data

	"""

	if len(data.shape) == 2:
		data = np.reshape(data, (1, data.shape[0], data.shape[1]))

	if isinstance(stride, int):
		stride = (stride, stride)

	# Calculate the ouptut size of each dimension.
	height_output_size = int(division_check(data.shape[1]-segment_height, stride[0]) + 1)
	width_output_size = int(division_check(data.shape[2]-segment_width, stride[1]) + 1)

	# Go through the data, getting each segment of the data that the layer will convolve over.
	h_index = 0
	w_index = 0

	for height in range(height_output_size):
		for width in range(width_output_size):

			# Go through batch size and then each of the 2 remaining dimensions.
			if len(data.shape) == 4:
				segment = data[0:, h_index:h_index+segment_height, w_index:w_index+segment_width, 0:]
			else:
				segment = data[0:, h_index:h_index+segment_height, w_index:w_index+segment_width]

			yield segment, height, width, h_index, w_index

			w_index += stride[1]
		h_index += stride[0]
		w_index = 0
