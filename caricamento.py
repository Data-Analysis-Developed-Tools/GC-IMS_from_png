import streamlit as st
from streamlit_cropper import st_cropper
from modules import data_loader  # Se il file è in una cartella "modules"
from components import data_loader  # Se il file è in una cartella "modules"
import estrazione_blob  # Importa il modulo di elaborazione

# Titolo dell'app
st.title("Segmentazione delle Macchie in Immagini GC-IMS")

# Caricare l'immagine usando il modulo data_loader
image = data_loader.load_image()

if image:
    # Interfaccia di cropping manuale
    st.subheader("Seleziona l'area da ritagliare")
    cropped_image = st_cropper(image, box_color='red', aspect_ratio=None)

    if cropped_image:
        st.subheader("Immagine Ritagliata")
        st.image(cropped_image, use_container_width=True)

        # Chiamata alla funzione di estrazione dei blob
        estrazione_blob.process_image(cropped_image)
