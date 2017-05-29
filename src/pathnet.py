# (c) Copyright 2017 Marc Assens. All Rights Reserved.

__author__ = "Marc Assens"
__version__ = "1.0"

import keras
keras.backend.set_image_dim_ordering("th")
from keras.models import load_model

from scipy import ndimage
import scipy.io as io
import numpy as np

import utils


def get_model():
	model = load_model('pathnet_model_v2.h5')
	return model


def predict(img_path):
	"""
		Predict 40 scanpaths given an image

		Params:
			image
	"""



	def sample_slice(pred, size, n_samples=1, use_heuristic=True, samples=[]):
	    """
	        Takes random points from the slice
	        taking into account the probabilities of each
	        pixel
	        
	        Params:
	            pred = 2d array with probability of each pixel
	            
	            size = (height, width) of the image
	            
	            n_samples = number of samples taken from this slice
	    """
	    
	    array_shape = (300, 600) 
	    
	    def num_to_pos(n, size):
	        """
	            Convert an index of the flatten 2d image to a 
	            coordenate (x, y)
	            
	            Params: 
	                size = (height, width)
	        """
	        # x and y pos
	        x = n % array_shape[1]
	        y = n / array_shape[1]
	        return (x, y)
	    
	    # Normalize to predictions
	    p = pred / np.sum(pred)
	    # Flatten
	    p = p.flatten()

	    # Store the samples 
	    #samples = []
	    for i in range(n_samples):
	        if samples and use_heuristic:
	            gaussian = np.zeros((300, 600))
	            gaussian[samples[-1][1], samples[-1][0]] = 1
	            gaussian = ndimage.filters.gaussian_filter(gaussian, [200, 200])
	            p = pred / np.sum(pred)
	            p =  p * gaussian
	            p = p / np.sum(p)
	            p = p.flatten()
	            
	            
	        n = np.random.choice(array_shape[0] * array_shape[1], p=p)
	        pos = num_to_pos(n, size=size)
	        samples.append(pos)
	        
	    
	    return samples

	def sample_volume(vol, n_samples=24, size=(3000, 6000), use_heuristic=True):
	    # Choose how many samples take per slice
	    # i.e. 
	    #     n_slices = 8 , n_samples = 12  => samples_per_slice = [2, 2, 2, 2, 1, 1, 1, 1]   
	    n_slices = vol.shape[0]
	    samples_per_slice = [n_samples / n_slices] * n_slices
	    for i in range(n_samples % n_slices):
	        samples_per_slice[i] += 1
	        
	        
	    # Sample each slice of the volume
	    samples = []
	    for i, n_samples in enumerate(samples_per_slice):
	        samples = sample_slice(vol[i], size, n_samples=n_samples, use_heuristic=use_heuristic, samples=samples)
	    
	    # Normalize positions
	    array_shape = (300, 600)
	    for i in range(len(samples)):
	        x = int((float(samples[i][0]) / array_shape[1]) * size[1])
	        y = int((float(samples[i][1]) / array_shape[0]) * size[0])
	        samples[i] = (x, y)
	        
	    return samples



	# Load image 
	img, img_size = utils.load_image(img_path)
	# Load model and predict volume
	model = get_model()
	preds = model.predict(img)


	scanpaths = []
	for i in range(40):
	    n_samples = utils.get_number_fixations()
	    s = sample_volume(preds[0], n_samples=n_samples , size=img_size, use_heuristic=True)

	    for j in range(n_samples):
	        # [user, index, time, x, y]
	        t = utils.get_duration_fixation()
	        pos_1s = [i+1, j+1, t, s[j][0], s[j][1]]
	        scanpaths.append(pos_1s)

	# Generate a np.array to output
	scanpaths_array = np.array(scanpaths, dtype=np.float32)

	return scanpaths_array

def predict_and_save(imgs_path, ids, out_path):
	""" 
		Predicts multiple images and saves them in .mat format
		on an output path

		Param:
			img_path : path where the images are
			ids: list with image ids
			out_path: path where the .mat files will be saved

		i.e.:
			img_path = '/root/sharedfolder/360Salient/'
			ids =  [29, 31]
			out_path =  '/root/sharedfolder/360Salient/results/'
	"""

	# Preproces and load images

	paths = utils.paths_for_images(imgs_path, ids)

	for i, path in enumerate(paths):
		print('Working on image %d of %d' % (i+1, len(paths)))

		# Predict the scanpaths
		scanpaths = predict(path)

		# Turn into a float np.array
		scanpaths = np.array(scanpaths, dtype=np.float32)

		# Save in output folder
		name = 'pred_%d' % ids[i]
		io.savemat(out_path + '%s.mat' % name, {name: scanpaths})

		print('Saved scanpaths from image %s in file %s.mat' % (path, name))

	print('Done!')
	return True
