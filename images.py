import numpy as np
from scipy.misc import imsave

config = {}
config["epsilon"] = 1e-7

EPSILON = config["epsilon"]


def min_max_normalization(x, l_range = 0.0, r_range = 1.0, axis = None):

	assert (l_range < r_range), "l_range must be less than r_range"

	min_x = np.amin(x, axis = axis, keepdims = True)
	max_x = np.amax(x, axis = axis, keepdims = True)

	out = (x - min_x) / np.maximum(max_x - min_x, EPSILON)
	out = ((r_range - l_range) * out) + l_range

	return out

def min_max_normalization_multiple_images(x, l_range = 0.0, r_range = 1.0, axis = None):

	if isinstance(axis, tuple):
		for a in axis:
			assert (a > 0), "Axis is out of bound or First axis represent different images."
	else:
		assert (axis > 0), "Axis is out of bound or First axis represent different images."

	out = np.zeros_like(x)

	for i in range(x.shape[0]):
		out[i] = min_max_normalization(x[i], l_range = l_range, r_range = r_range, axis = axis - 1)

	return out

def mean_std_normalization(x, mean = 0.0, stddev = 1.0, axis = None):
	out = None

	mean_x = np.mean(x, axis = axis, keepdims = True)
	std_x = np.std(x, axis = axis, keepdims = True)

	out = (x - mean_x) / np.maximum(std_x, EPSILON)
	out = (out * stddev) + mean

	return out

def mean_std_normalization_multiple_images(x, mean = 0.0, stddev = 1.0, axis = None):

	if isinstance(axis, tuple):
		for a in axis:
			assert (a > 0), "Axis is out of bound or First axis represent different images."
	else:
		assert (axis > 0), "Axis is out of bound or First axis represent different images."

	out = np.zeros_like(x)

	for i in range(x.shape[0]):
		out[i] = mean_std_normalization(x[i], mean = mean, stddev = stddev, axis = axis - 1)

	return out

def clip_pixel_value(x, l_bound = 0.0, r_bound = 1.0):
	return np.clip(x, a_min = l_bound, a_max = r_bound)

def save_image(image, path, fmt = None):
	is_success = True

	assert (len(image.shape) in [2, 3]), "image argument must be 2-D or 3-D"
	if len(image.shape) == 3:
		assert (image.shape[-1] in [3, 4]), "if image argument is 3-D then last dimension must be 3 or 4"

	if not os.path.exists(path):
		try:
			os.mkdir(os.path.dirname(path))
		except:
			raise("Error in making directory. Try with sudo.")

	try:
		imsave(path, image, format = fmt)
	except:
		is_success = False

	return is_success

def save_images(images, dirpath, file_name, fmt = None):
	images_saved = 0

	assert (images.shape[0] < len(file_name)), "Please provide enough image file names."

	for i in range(images.shape[0]):
		if save_image(images[i], path = os.path.join(dirpath, file_name[i]), fmt = fmt):
			images_saved += 1

	return images_saved
