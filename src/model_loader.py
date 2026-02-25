import gdown
import os
import tensorflow as tf
import streamlit as st

@st.cache_resource
def load_model():

    model_path = "model/cat_dog_model.h5"

    if not os.path.exists(model_path):

        os.makedirs("model", exist_ok=True)

        gdown.download(id="1cAx5OawpOkuxV0Ql5pRT49QoHncN7f8k", output=model_path, quiet=False)

    return tf.keras.models.load_model(model_path)