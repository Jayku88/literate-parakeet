import tensorflow as tf
model = tf.keras.models.load_model('newmodel1_feb_2.h5')
import streamlit as st
st.write("""
         # Rock-Paper-Scissor Hand Sign Prediction
         """
         )
st.write("This is a simple image classification web app to predict rock-paper-scissor hand sign")
model.summary(print_fn=lambda x: st.text(x))
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np


def import_and_predict(image_data, model):
    size = (224,224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = np.expand_dims(image, axis=0)
    prediction = model.predict(img)

    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)

    if np.argmax(prediction) == 0:
        st.write("It is a paper!")
    elif np.argmax(prediction) == 1:
        st.write("It is a rock!")
    else:
        st.write("It is a scissor!")

    st.text("Probability (0: Paper, 1: Rock, 2: Scissor")
    st.write(prediction)