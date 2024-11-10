import streamlit as st
import pickle
import numpy as np
import json

# Load the model and columns configuration
with open("columns (1).json", "r") as f:
    data_columns = json.load(f)
    location_columns = data_columns["location_columns"]
    area_columns = data_columns["area_columns"]
    availability_columns = data_columns["availability_columns"]

# Load the trained model
model = pickle.load(open("banglore_home_prices_model (1).pickle", "rb"))

def predict_house_price(location, area, availability, sqft, bhk, bathrooms):
    try:
        # Find index of location, area, availability columns
        loc_index = location_columns.index(location)
        area_index = area_columns.index(area)
        availability_index = availability_columns.index(availability)
    except ValueError:
        loc_index = area_index = availability_index = -1

    # Initialize arrays for one-hot encoding
    loc_array = np.zeros(len(location_columns))
    area_array = np.zeros(len(area_columns))
    availability_array = np.zeros(len(availability_columns))

    if loc_index >= 0:
        loc_array[loc_index] = 1
    if area_index >= 0:
        area_array[area_index] = 1
    if availability_index >= 0:
        availability_array[availability_index] = 1

    # Combine features into a single array
    sample = np.concatenate(
        (np.array([sqft, bhk, bathrooms]), availability_array[:-1], area_array[:-1], loc_array[:-1])
    )
    # Return predicted price
    return model.predict([sample])[0]

# Streamlit App
st.title("Bangalore House Price Prediction")

# Input fields
sqft = st.number_input("Total Square Feet", min_value=500, max_value=5000, step=10)
bhk = st.number_input("BHK", min_value=1, max_value=10, step=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)
location = st.selectbox("Location", location_columns)  # Populate from columns.json
area = st.selectbox("Area Type", area_columns)  # Populate from columns.json
availability = st.selectbox("Availability", availability_columns)  # Populate from columns.json

# Predict the price when button is clicked
if st.button("Predict Price"):
    if location and area and availability:
        price = predict_house_price(location, area, availability, sqft, bhk, bathrooms)
        st.write(f"Estimated House Price: ₹{price} Lakhs")
    else:
        st.write("Please fill all the fields")
