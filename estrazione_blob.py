import cv2
import numpy as np
import io
from PIL import Image
import streamlit as st

def is_monotonic_gradient(region):
    """Verifica se un blob ha un unico massimo di intensità."""
    min_val, max_val, _, max_loc = cv2.minMaxLoc(region)
    return (region == max_val).sum() == 1

def adjust_bounding_box(x, y, w, h, x_max, y_max, img_shape):
    """Riposiziona il bounding box per centrare il massimo cromatico nella zona centrale."""
    H, W = img_shape  # Altezza e larghezza dell'immagine

    # Calcola i quintili centrali
    x_quintile = (w // 5)  
    y_quintile = (h // 5)

    x_start, x_end = x, x + w
    y_start, y_end = y, y + h

    # Controlla se il massimo è troppo vicino ai bordi
    if x_max < x + x_quintile:
        x_start = max(0, x_max - x_quintile)
        x_end = min(W, x_start + w)
    elif x_max > x + (4 * x_quintile):
        x_end = min(W, x_max + x_quintile)
        x_start = max(0, x_end - w)

    if y_max < y + y_quintile:
        y_start = max(0, y_max - y_quintile)
        y_end = min(H, y_start + h)
    elif y_max > y + (4 * y_quintile):
        y_end = min(H, y_max + y_quintile)
        y_start = max(0, y_end - h)

    return x_start, y_start, x_end - x_start, y_end - y_start

def process_image(image):
    """Elabora l'immagine ritagliata per individuare i blob e visualizzarli in una gallery con centratura migliorata."""
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
    st.image(thresh, use_container_width=True, caption="Macchie Segmentate")

    blob_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img_gray[y:y+h, x:x+w])
        x_max, y_max = x + max_loc[0], y + max_loc[1]

        # Riposiziona il bounding box se necessario
        x, y, w, h = adjust_bounding_box(x, y, w, h, x_max, y_max, img_gray.shape)

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
            st.image(blob_img, caption=f"Blob {i+1}", use_container_width=True)
