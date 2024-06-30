import os
import json
import requests

import pandas as pd
import streamlit as st

from src.preprocess import Preprocessor

DATA_PATH = os.getenv("DATA_PATH", default="data/titanic-training-data.csv")
COLUMN_CONFIG = os.getenv("COLUMN_CONFIG", default="config/columns.json")
EMBARKED_CONFIG = os.getenv("EMBARKED_CONFIG", default="config/embarked.json")
TITLE_CONFIG = os.getenv("TITLE_CONFIG", default="config/title.json")

st.title("Titanic Survival Prediction")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH).drop(columns=["PassengerId"])
data = load_data()

@st.cache_data
def load_columns():
    with open(COLUMN_CONFIG, "r") as f:
        return json.load(f)
columns = load_columns()

@st.cache_data
def load_embarked():
    with open(EMBARKED_CONFIG, "r") as f:
        return json.load(f)
embarked = load_embarked()

@st.cache_data
def load_title():
    with open(TITLE_CONFIG, "r") as f:
        return json.load(f)
title = load_title()

# Convert the JSON payload to a DataFrame

# Use the loaded model to make a prediction
# prediction = loaded_model.predict(data_df)
# Return the prediction as a JSON response
# st.write(prediction.tolist())

with st.sidebar:
    st.write("### Passenger Configuration") 

    # Receive the Passenger's class from a dropdown
    passenger_class = st.selectbox("Class", data["Pclass"].dropna().unique())

    # Receive the Passenger's Title from a dropdown
    passenger_title = st.selectbox("Title", title.keys())

    # Receive the Passenger's Age from a positive number input
    passenger_age = st.number_input("Age", min_value=0)

    # Receive the Passenger's SibSp from a positive number input
    passenger_sibsp = st.number_input("SibSp", min_value=0)

    # Receive the Passenger's Parch from a positive number input
    passenger_parch = st.number_input("Parch", min_value=0)

    # Receive the Passenger's Fare from a positive number input
    passenger_fare = st.number_input("Fare", min_value=0)

    # Receive the Passenger's Embarked from a dropdown
    passenger_embarked = st.selectbox("Embarked", data["Embarked"].dropna().unique())

    # Receive the Passenger's Cabin from a dropdown
    passenger_cabin = st.selectbox("Cabin", data["Cabin"].dropna().unique())

    # Receive Passenger's Sex from a dropdown
    passenger_sex = st.selectbox("Sex", data["Sex"].dropna().unique())

    # Receive Passenger's Ticket from a dropdown
    passenger_ticket = st.selectbox("Ticket", data["Ticket"].dropna().unique())

    payload = {
        "Pclass": [passenger_class],
        "Title": [passenger_title],
        "Sex": [passenger_sex],
        "Age": [passenger_age],
        "SibSp": [passenger_sibsp],
        "Parch": [passenger_parch],
        "Ticket": [passenger_ticket],
        "Fare": [passenger_fare],
        "Cabin": [passenger_cabin],
        "Embarked": [passenger_embarked],
    }

    # Convert the input to a DataFrame
    df = pd.DataFrame(payload)

st.write("### Selected Passenger Configuration")
st.write(df)

# Send the JSON payload to the /predict endpoint
st.write("### JSON Payload")
st.json(payload, expanded=False)

if st.button("Predict"):
    st.spinner("Making Prediction...")
    response = requests.post("http://localhost:5000/predict", json=df.to_dict())
    if response.status_code == 200:
        prediction = response.json()[0]
        if prediction == 1:
            st.write("### Prediction")
            st.write(f"Survived: {prediction} (`Yes`) 🎉")
        else:
            st.write("### Prediction")
            st.write(f"Survived: {prediction} (`No`) 😢")
        st.success("Prediction made successfully!")
        st.balloons()
    else:
        st.write("Failed to make prediction")
        st.write(response)
