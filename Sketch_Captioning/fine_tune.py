import os, glob
import os.path
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Convolution3D, MaxPooling2D
from keras.models import load_model
from os import listdir
from pickle import dump
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import keras

print(keras.__version__)
# extract features from each photo in the directory
def extract_features(directory):
	# load the model
	#model = load_model('sketch_inceptionv3.h5')
	model = InceptionV3()
	# re-structure the model
	model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
	# summarize
	print(model.summary())
	# extract features from each photo
	features = dict()
	for name in listdir(directory):
		# load an image from file
		filename = directory + '/' + name
		image = load_img(filename, target_size=(299, 299))
		# convert the image pixels to a numpy array
		image = img_to_array(image)
		# reshape data for the model
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepare the image for the VGG model
		image = preprocess_input(image)
		# get features
		feature = model.predict(image, verbose=0)
		# get image id
		image_id = name.split('.')[0]
		# store feature
		features[image_id] = feature
		print('>%s' % name)
	return features


# extract features from all images
directory = 'Sketch_distancetransform' #dataset 새로운거
features = extract_features(directory)
print('Extracted Features: %d' % len(features))
# save to file
dump(features, open('features_dt.pkl', 'wb'))