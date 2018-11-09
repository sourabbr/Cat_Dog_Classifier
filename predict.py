#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 17:53:13 2018

@author: sourab
"""

from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/dog.jpeg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
# load json and create model
json_file = open('catdogtrained.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
# load weights into new model
classifier.load_weights("catdogtrained.h5")
print("Loaded model from disk")
result = classifier.predict(test_image)
print(result)
#training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)