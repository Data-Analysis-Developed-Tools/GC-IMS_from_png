import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper
import estrazione_blob  # Importa il modulo esterno

# Titolo dell'app
st.title("Segmentazione delle Macchie in Immagini GC-IMS")

# Caricamento immagine
uploaded_file = st.file_uploader("Carica un'immagine (.png)", type=["png"])

if uploaded_file:
    # Apertura dell'immagine con PIL
    image = Image.open(uploaded_file)
    
    # Interfaccia di cropping manuale
    st.subheader("Seleziona l'area da ritagliare")
    cropped_image = st_cropper(image, box_color='red', aspect_ratio=None)

    if cropped_image:
        st.subheader("Immagine Ritagliata")
        st.image(cropped_image, use_container_width=True)

        # Chiamata alla funzione di estrazione dei blob
        estrazione_blob.process_image(cropped_image)
