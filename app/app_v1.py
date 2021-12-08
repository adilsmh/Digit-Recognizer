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
st.header('Pick a random image from test set')

SIZE = 400

# define stats model performance variables in session-state object
# stats_vars = ['TP', 'Num_of_predictions']
#
# for i in stats_vars:
#     if i not in st.session_state:
#         st.session_state.i = 0

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
