# -*- coding: utf-8 -*-

import numpy as np
import pickle
import sys


try:
    # loading the saved model
    loaded_model = pickle.load(open("C:/Users/sneha/OneDrive/Desktop/Deployed ML Models/trained_model1.sav", 'rb'))
except FileNotFoundError:
    print("ERROR: Model file not found at the specified path")
    sys.exit(1)
except Exception as e:
    print(f"ERROR loading model: {e}")
    sys.exit(1)

# get the loaded model
input = (5,166,72,19,175,25.8,0.587,51)

# changing the data into a nparray
input_array = np.asarray(input)

# reshaped the data
input_reshaped = input_array.reshape(1, -1)

# checking the model
prediction = loaded_model.predict(input_reshaped)
print(prediction)
if(prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')