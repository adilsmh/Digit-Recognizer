# import needed libraries
import time

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

# define stats model performance variables in session-state object
if 'num_of_pred' not in st.session_state:
    st.session_state.num_of_pred = 0
if 'tp' not in st.session_state:
    st.session_state.tp = 0
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = 0

st.title('Handwritten digit recognition')
st.subheader('Choose a DRM version between those below')

version_option = st.selectbox('', ('Select here', 'V1', 'V2'))

if version_option == 'V1':
    st.header('Pick a random image')

    SIZE = 400

    image = ''

    if st.button('Pick/predict random image'):
        image = df_test.values[rnd.randrange(df_test.shape[0])].reshape(-1, 28, 28, 1)
        st.image(image, width=SIZE)
        # instantiate model fot prediction
        val = model.predict(image)
        # progress bar (fake)
        with st.spinner(text='In progress'):
            st.success('Done')
        # return predicted digit
        st.write(f'result: {np.argmax(val[0])}')

elif version_option == 'V2':
    st.header('Draw an digit between 0-9')

    SIZE = 400

    col1, col2 = st.columns([3, 2])
    
    with col1:
        canvas_result = st_canvas(
            fill_color='#000000',
            stroke_width=20,
            stroke_color='#FFFFFF',
            background_color='#000000',
            width=SIZE,
            height=SIZE,
            key='canvas')

    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))    
        
    # define increment count functions
    def increment_tp():
        st.session_state.tp += 1
    def increment_num_pred():
        st.session_state.num_of_pred += 1
    def calc_accuracy():
        # return model accuracy
        st.session_state.accuracy = ((st.session_state.tp / st.session_state.num_of_pred) * 100)
        st.write(f'num of predictions: {st.session_state.accuracy}')

    val = ''

    def predict():
        # instantiate model for prediction
        test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # instantiate model for prediction
        val = model.predict(test_x.reshape(1, 28, 28))
        # progress bar (viz only)
        with st.spinner('Wait for it...'):
            time.sleep(3)
        st.success('Done!')
        # inject num of pred count
        st.session_state.num_of_pred += 1
        # return predicted digit / num of predictions
        st.write(f'result: {np.argmax(val[0])}')

        st.button('YES', on_click=good_predict)
        st.button('NO', on_click=bad_predict)

    def good_predict():
        st.session_state.tp += 1
        tp = st.session_state.tp
        num_of_pred = st.session_state.num_of_pred
        st.session_state.accuracy = tp / num_of_pred
        accuracy = st.session_state.accuracy
        st.write(f'Accuracy score: {accuracy}')
    def bad_predict():
        tp = st.session_state.tp
        num_of_pred = st.session_state.num_of_pred
        st.session_state.accuracy = tp / num_of_pred
        accuracy = st.session_state.accuracy
        st.write(f'Accuracy score: {accuracy}')


    st.button('Predict', on_click=predict)









