import numpy as np

# Goes through a list of the args and kwargs of a function and compares them to the expected data types.
def accepts(**types):
	def check_accepts(func):
		assert len(types) == func.__code__.co_argcount,\
			   '@accept number of arguments not equal with function number of arguments in "%s"' % func.__name__

		def new_func(*args, **kwargs):
			for i, value in enumerate(args):
				var_name = func.__code__.co_varnames[i]

				if var_name in types and types[var_name] == "any":
					continue

				elif var_name in types and not isinstance(value, types[var_name]):
					raise ValueError("At %s(): arg '%s' expected type '%s', but recieved type '%s'"\
									 % (func.__name__, var_name, types[var_name], type(value)))

			for key, value in kwargs.items():
				if key in types and types[key] == "any":
					continue

				elif key in types and not isinstance(value, types[key]):
					raise ValueError("At %s(): arg '%s' expected type '%s', but recieved type '%s'"\
									 % (func.__name__, key, types[key], type(value)))

			return func(*args, **kwargs)
		new_func.__name__ = func.__name__
		return new_func
	return check_accepts

# Divides np.ndarrays or floats and catches division errors.
@accepts(arg1=(np.ndarray, float, int), divisor=(np.ndarray, float, int), new_divide=float)
def division_check(arg1, divisor, new_divide=1.0e-8) -> (np.ndarray, float):
	if isinstance(divisor, np.ndarray):
		new_divisor = np.where(divisor == 0, new_divide, divisor)
		return arg1/new_divisor

	else:
		if divisor == 0:
			return arg1/new_divide
		else:
			return arg1/divisor
