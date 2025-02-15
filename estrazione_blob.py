import cv2
import numpy as np
import io
from PIL import Image
import streamlit as st

def is_monotonic_gradient(region):
    """Verifica se un blob ha un unico massimo di intensità."""
    min_val, max_val, _, max_loc = cv2.minMaxLoc(region)
    return (region == max_val).sum() == 1

def process_image(image):
    """Elabora l'immagine ritagliata per individuare i blob e visualizzarli in una gallery."""
    img_np = np.array(image.convert("RGB"))
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Cursore per la soglia cromatica
    threshold_value = st.slider("Soglia per la segmentazione", 0, 255, 150)

    # Applicare un threshold
    _, thresh = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)

    # Trovare i contorni
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Visualizzare l'immagine segmentata
    st.subheader("Immagine Segmentata")
    st.image(thresh, use_column_width=True, caption="Macchie Segmentate")

    blob_images = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cropped_blob = img_np[y:y+h, x:x+w]

        # Verifica se il blob ha un unico massimo o un gradiente monotono
        if is_monotonic_gradient(img_gray[y:y+h, x:x+w]):
            blob_pil = Image.fromarray(cropped_blob)
            blob_images.append(blob_pil)
        else:
            # Se il gradiente non è monotono, suddividere in due parti
            half_h = h // 2
            sub_regions = [(y, y+half_h), (y+half_h, y+h)]
            for sub_y_start, sub_y_end in sub_regions:
                cropped_sub_blob = img_np[sub_y_start:sub_y_end, x:x+w]
                blob_pil = Image.fromarray(cropped_sub_blob)
                blob_images.append(blob_pil)

    # Mostrare i blob ritagliati in una griglia a 5 colonne
    st.subheader("Galleria di Blob Identificati")
    cols = st.columns(5)  # Creiamo 5 colonne

    for i, blob_img in enumerate(blob_images):
        with cols[i % 5]:  # Distribuisce le immagini nelle 5 colonne
            st.image(blob_img, caption=f"Blob {i+1}", use_column_width=True)
