import pathlib
import streamlit as st
from fastai.vision.all import *

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

# setting title, app title and favicon
st.set_page_config(page_title='Ball Classifier', page_icon='balls.jpg',
 layout="centered")
st.title('A model for classifying football, tennis and golf balls')

# creating buttons 
file = st.file_uploader('Add file', type=['tif','tiff','bmp','jpg','jpeg','gif','png','svg'])
c1, c2, c3 = st.columns((3,1,3))
with c2: cl_button = st.button('Classify')
# loading a classification model
cl_model = load_learner('ball_classifier.pkl')

# outputting the prediction
def on_click_classifier():
    if file:
        img = PILImage.create(file)
        pred, pred_idx, probs = cl_model.predict(img)
        col1, col2, col3 = st.columns((4,1.5,1))
        with col1: st.image(img)
        with col2: 
            st.success(f"Prediction: {pred}")
            st.info(f"Probability: {probs[pred_idx]:.4f}")
    else:
        st.error('Upload a picture for classification')
if cl_button: on_click_classifier()
