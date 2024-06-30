import os
import json
import requests
import base64

import pandas as pd
import streamlit as st

from src.preprocess import Preprocessor

DATA_PATH = os.getenv("DATA_PATH", default="data/titanic-training-data.csv")
COLUMN_CONFIG = os.getenv("COLUMN_CONFIG", default="config/columns.json")
EMBARKED_CONFIG = os.getenv("EMBARKED_CONFIG", default="config/embarked.json")
TITLE_CONFIG = os.getenv("TITLE_CONFIG", default="config/title.json")

BG_IMAGE_PATH = "assets/imgs/titanic.png"

# Set the background imaged for the app
@st.cache_resource
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file, opacity=0.5):
    # Encode the binary file to base64
    bin_str = get_base64_of_bin_file(png_file)
    # CSS to set the background image with reduced contrast using an overlay
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(255, 255, 255, {opacity}), rgba(255, 255, 255, {opacity})), url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}
    </style>
    '''
    
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_png_as_page_bg(BG_IMAGE_PATH, opacity=0.7)

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

# Set a box area to display the selected passenger configuration

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
            st.markdown(f":grey-background[:blue[*Survived* **{prediction}** ***(`Yes`) 🎉***]]")
        else:
            st.write("### Prediction")
            st.markdown(f":grey-background[:red[*Survived* **{prediction}** ***(No) 😢***]]")
        st.success("Prediction made successfully!")
        st.balloons()
    else:
        st.error("Failed to make prediction")
        st.write(response.content)

#------------------------------------------------------------------------------------------------------------
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb
import streamlit as st

def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))

def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)

def layout(*args):
    style = """
    <style>
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 50px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        display="flex",
        justify_content="space-between",
        align_items="flex-end",  # Align content to the bottom
        padding=px(10, 20),
        color="black",
        text_align="center",
        height="auto",
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(0, "auto"),
        border_style="none",
        border_width=px(0.5),
        color='rgba(0,0,0,.5)'
    )

    body = div(style=styles(display="flex", justify_content="space-between", width=percent(100)))()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            # Only for string text directly, we don't have it now
            pass
        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        div()(
            "Background image credits: ",
            link("https://pixabay.com/users/iffany-6128830/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=8738962", "Ivana Tomášková", color="#4682B4"),
            " from ",
            link("https://pixabay.com//?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=8738962", "Pixabay", color="#4682B4"),
        ),
        div()(
            "Designed by ",
            link("https://linkedin.com/in/joshua-olalemi/", "Joshua Olalemi", color="#4682B4"),
        )
    ]
    layout(*myargs)

if __name__ == "__main__":
    footer()

#------------------------------------------------------------------------------------------------------------
