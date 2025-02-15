import streamlit as st
from streamlit_cropper import st_cropper
import data_loader  # Importa il modulo per il caricamento dell'immagine
import estrazione_blob  # Importa il modulo di elaborazione

# Titolo dell'app
st.title("Segmentazione delle Macchie in Immagini GC-IMS")

# Caricare l'immagine dalla barra laterale
image = data_loader.load_image()

if image:
    st.sidebar.success("✅ Immagine caricata con successo!")
    
    # Interfaccia di cropping manuale
    st.subheader("Seleziona l'area da ritagliare")
    cropped_image = st_cropper(image, box_color='red', aspect_ratio=None)

    if cropped_image:
        st.subheader("Immagine Ritagliata")
        st.image(cropped_image, use_container_width=True)

        # Chiamata alla funzione di estrazione dei blob
        estrazione_blob.process_image(cropped_image)
else:
    st.sidebar.warning("⚠️ Nessuna immagine caricata. Carica un file PNG per continuare.")
