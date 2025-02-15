import cv2
import numpy as np
import io
from PIL import Image
import streamlit as st

def is_monotonic_gradient(region):
    """
    Controlla se il gradiente cromatico è monotono, ovvero se l'intensità decresce 
    allontanandosi dal massimo centrale in tutte le direzioni.
    """
    min_val, max_val, _, max_loc = cv2.minMaxLoc(region)
    x_max, y_max = max_loc  # Coordinate del massimo cromatico

    # Convertiamo l'immagine in float per calcolare i gradienti
    region = region.astype(np.float32)

    # Estrai i gradienti lungo x e y
    grad_x = np.diff(region, axis=1)  # Differenze tra colonne
    grad_y = np.diff(region, axis=0)  # Differenze tra righe

    # Verifica che il gradiente sia negativo (decrescente) allontanandosi dal massimo
    left = np.all(grad_x[:, :x_max] <= 0)  # Sinistra
    right = np.all(grad_x[:, x_max:] >= 0)  # Destra
    top = np.all(grad_y[:y_max, :] <= 0)  # Sopra
    bottom = np.all(grad_y[y_max:, :] >= 0)  # Sotto

    return left and right and top and bottom  # Deve essere vero per tutte le direzioni

def adjust_bounding_box(x, y, w, h, x_max, y_max, img_gray):
    """
    Regola il bounding box per assicurarsi che il massimo cromatico sia centrato e
    che il gradiente sia monotono decrescente rispetto al massimo.
    """
    H, W = img_gray.shape  # Dimensioni totali dell'immagine

    # Espansione massima consentita
    max_expand = 50  # Pixel di espansione per migliorare il centraggio
    for _ in range(max_expand):
        roi = img_gray[y:y+h, x:x+w]
        if is_monotonic_gradient(roi):
            return x, y, w, h  # Bounding box già ottimale

        # Se il gradiente non è valido, espandi la finestra di 1 pixel in ogni direzione (se possibile)
        if x > 0: x -= 1
        if y > 0: y -= 1
        if x + w < W: w += 1
        if y + h < H: h += 1

    return x, y, w, h  # Restituisce il riquadro corretto o il migliore ottenibile

def process_image(image):
    """Elabora l'immagine ritagliata per individuare i blob e visualizzarli in una gallery ottimizzata."""
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

        # Regola il bounding box per ottimizzare il centraggio e il gradiente cromatico
        x, y, w, h = adjust_bounding_box(x, y, w, h, x_max, y_max, img_gray)
        cropped_blob = img_np[y:y+h, x:x+w]

        blob_pil = Image.fromarray(cropped_blob)
        blob_images.append(blob_pil)

    # Mostrare i blob ritagliati in una griglia a 5 colonne
    st.subheader("Galleria di Blob Identificati")
    cols = st.columns(5)  # Creiamo 5 colonne

    for i, blob_img in enumerate(blob_images):
        with cols[i % 5]:  # Distribuisce le immagini nelle 5 colonne
            st.image(blob_img, caption=f"Blob {i+1}", use_container_width=True)
