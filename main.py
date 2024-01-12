## Import Libraries
import joblib
import streamlit as st
import numpy as np
from utlis import process_new
import pandas as pd
df=pd.read_excel("Data_Train.xlsx")

## Load the model
model = joblib.load('model.pkl')




def Air_linePrice_regression():

    ## Title
    st.title('Air line Price Prediction ....')
    st.markdown('<hr>', unsafe_allow_html=True)



    ## Input fields

    Airline = st.selectbox('Airline', options=df["Airline"].unique().tolist())
    Source = st.selectbox('Source', options=df["Source"].unique().tolist())
    Destination = st.selectbox('Destination', options=df["Destination"].unique().tolist())
    Total_Stops = st.selectbox('Total_Stops', options=[2, 1, 0, 3, 4])
    Day_of_Journey = st.selectbox('Day_of_Journey', options=['Sunday', 'Tuesday', 'Friday', 'Thursday', 'Wednesday', 'Monday', 'Saturday'])
    Month_of_Journey = st.selectbox('Month_of_Journey', options=['June', 'March', 'May', 'April'])
    Dep_Time_Hour = st.selectbox('Airline_Time_Hour', options=['Am', 'Pm'])
    Duration_with_minutes = st.slider('Airline Duration with minutes ',85,2860,10)
    st.markdown('<hr>', unsafe_allow_html=True)


    if st.button('Predict Pirce ...'):

        ## Concatenate the users data
        new_data = np.array([Airline, Source, Destination, Total_Stops, Day_of_Journey,
                            Month_of_Journey, Dep_Time_Hour, Duration_with_minutes])
        
        ## Call the function from utils.py to apply the pipeline
        X_processed = process_new(X_new=new_data)

        ## Predict using Model
        
        y_pred = model.predict(X_processed)


        y_pred = np.exp(y_pred) - 1

        ## Display Results
        st.success(f'Air line Price Prediction is ... {y_pred}')



if __name__ == '__main__':
    ## Call the function
    Air_linePrice_regression()

