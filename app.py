import streamlit as st
import numpy as np
import pandas as pd
import pickle

pipe = pickle.load(open('pipe.pkl','rb'))

st.title('Car Price Predictor')

#year
year = st.number_input('Make Year')
#km driven
km = st.number_input('KMs Driven')
#fuel
fuel = st.selectbox('Fuel Type',('Diesel','Petrol'))
#sellertype
seller = st.selectbox('Seller Type',('Individual','Dealer'))
#transmission
transmission = st.selectbox('Transmission',('Manual','Automatic'))
#owner
owner = st.selectbox('Owner',('First Owner','Second Owner','Third Owner'))
#mileage
mileage = st.number_input('Mileage')
#engine
engine = st.number_input('Engine')
#max power
maxpower = st.number_input('Max Power')
#seats
seats = st.number_input('Seats')
#brand
brand = st.selectbox('Brand',('Maruti','Hyundai','Mahindra','Tata','Ford','Honda','Toyota','Chevrolet','Renault','Volkswagen'))

if st.button('Predict Price'):
    input = np.array([[year,km,fuel,seller,transmission,owner,mileage,engine,maxpower,seats,brand]])
    input = pd.DataFrame(input,columns=['year','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats','brand'])
    y_pred = pipe.predict(input)
    st.title("Rs " + str(np.round(y_pred[0])))