# import needed libraries
import random as rnd
import pandas as pd
import numpy as np
import matplotlib as plt
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas

df_test = pd.read_csv('data/test/raw/test.csv')
model = load_model('models/model_v1.h5')

st.title('Handwritten digit recognition')
st.header('Choose a DRM version between those below')

SIZE = 400

canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    key='canvas')

# define stats model performance variables in session-state object
if 'tp' not in st.session_state:
    st.session_state.tp = 0
if 'tn' not in st.session_state:
    st.session_state.tn = 0
if 'num_of_predictions' not in st.session_state:
    st.session_state.num_pred = 0
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = 0

# define increment count functions
def increment_tp():
    st.session_state.tp += 1
def increment_tn():
    st.session_state.tn += 1
def increment_num_pred():
    st.session_state.num_pred += 1
def calc_accuracy():
    # return model accuracy
    st.session_state.accuracy = ((st.session_state.tp / st.session_state.num_pred) * 100)
    st.write(f'num of predictions: {st.session_state.accuracy}')

# plot input refactored image
img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
st.write('Model Input')
st.image(rescaled)

val = ''

if st.button('Predict'):
    # instantiate model for prediction
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # instantiate model for prediction
    val = model.predict(test_x.reshape(1, 28, 28))
    # inject num of predictions count
    increment_num_pred()
    # return predicted digit / num of predictions
    st.write(f'result: {np.argmax(val[0])}')
    st.write(f'num of predictions: {st.session_state.num_pred}')
    # inject tp/tn count
    if st.button("YES", on_click=increment_tp()):
        calc_accuracy()
    if st.button("NO", on_click=increment_tn()):
        calc_accuracy()










