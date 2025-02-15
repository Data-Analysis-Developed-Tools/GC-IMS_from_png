import streamlit as st
from PIL import Image

def load_image():
    """Carica un'immagine PNG e restituisce un oggetto PIL.Image."""
    uploaded_file = st.file_uploader("Carica un'immagine (.png)", type=["png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        return image
    return None
