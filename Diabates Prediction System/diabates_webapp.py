# -*- coding: utf-8 -*-

import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open("C:/Users/sneha/OneDrive/Desktop/Deployed ML Models/trained_model1.sav", 'rb'))

# creating a function to predict

def diabates_prediction(input):
    
    # changing the data into a nparray
    input_array = np.asarray(input)

    # reshaped the data
    input_reshaped = input_array.reshape(1, -1)

    # checking the model
    prediction = loaded_model.predict(input_reshaped)
    # print(prediction)
    
    if(prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
# main function
def main():
    
    # giving a title
    st.title("Diabates Prediction")
    
    # getting the input data from the user
    
    Pregnancies = st.text_input("Enter the number of pregnancies:")	
    Glucose	= st.text_input("Enter the glucose level::")	
    BloodPressure =	st.text_input("Enter the bloodpressure:")	
    SkinThickness = st.text_input("Enter the skinthickness:")	
    Insulin = st.text_input("Enter the blood insulin level:")
    BMI = st.text_input("Enter the BMI:")
    DiabetesPedigreeFunction = st.text_input("Enter the number:")
    Age = st.text_input("Enter the age:")
    
    # code for prediction
    diagonosis = ''
    
    # creating a button
    
    if st.button('Diabates test result'):
        diagonosis = diabates_prediction([Pregnancies, Glucose,	BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
    
    st.success(diagonosis)
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    