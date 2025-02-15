import cv2
import numpy as np
import io
from PIL import Image
import streamlit as st

def find_maxima(region, threshold=5):
    """
    Trova tutti i massimi locali nell'immagine e restituisce le loro coordinate.
    threshold: differenza minima dal massimo globale per considerare un massimo.
    """
    min_val, max_val, _, _ = cv2.minMaxLoc(region)
    
    # Trova tutti i punti con valore vicino al massimo globale
    max_mask = (region >= max_val - threshold).astype(np.uint8) * 255

    # Trova i contorni dei punti di massimo
    contours, _ = cv2.findContours(max_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    maxima_points = [cv2.minMaxLoc(region[y:y+1, x:x+1])[3] for c in contours for x, y in c[:, 0]]
    
    return maxima_points if len(maxima_points) > 1 else None

def split_blob(x, y, w, h, maxima):
    """
    Divide il blob in due sotto-blob in base alla posizione dei massimi.
    Se i massimi sono più distanziati in verticale, divide orizzontalmente.
    Se sono più distanziati in orizzontale, divide verticalmente.
    """
    x_max1, y_max1 = maxima[0]
    x_max2, y_max2 = maxima[1]

    if abs(x_max1 - x_max2) > abs(y_max1 - y_max2):
        x_mid = (x_max1 + x_max2) // 2
        return [(x, y, x_mid - x, h), (x_mid, y, x + w - x_mid, h)]
    else:
        y_mid = (y_max1 + y_max2) // 2
        return [(x, y, w, y_mid - y), (x, y_mid, w, y + h - y_mid)]

def process_image(image):
    """Elabora l'immagine ritagliata per individuare i blob e segmentarli iterativamente."""
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

    queue = [(cv2.boundingRect(contour)) for contour in contours]  # Code di elaborazione
    blob_images = []

    while queue:
        x, y, w, h = queue.pop(0)  # Estrai il primo riquadro dalla coda
        region = img_gray[y:y+h, x:x+w]

        # Controlla se ci sono più massimi cromatici
        maxima = find_maxima(region)

        if maxima and len(maxima) > 1:
            # Se ci sono due massimi, suddividi il blob in due
            new_bounding_boxes = split_blob(x, y, w, h, maxima)
            queue.extend(new_bounding_boxes)  # Aggiungi le nuove regioni alla coda
        else:
            # Se c'è un solo massimo, mantieni il bounding box attuale
            cropped_blob = img_np[y:y+h, x:x+w]

            # Assicurarsi che l'immagine sia sempre in formato RGB e uint8
            if len(cropped_blob.shape) == 2:
                cropped_blob = cv2.cvtColor(cropped_blob, cv2.COLOR_GRAY2RGB)

            if cropped_blob.dtype != np.uint8:
                cropped_blob = (cropped_blob * 255).astype(np.uint8)

            blob_pil = Image.fromarray(cropped_blob, mode="RGB")

            # Convertire PIL Image in PNG Bytes prima di visualizzarla
            buf = io.BytesIO()
            blob_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()

            blob_images.append(byte_im)

    # Mostrare i blob ritagliati in una griglia a 5 colonne
    st.subheader("Galleria di Blob Identificati")
    cols = st.columns(5)  # Creiamo 5 colonne

    for i, blob_img in enumerate(blob_images):
        with cols[i % 5]:  # Distribuisce le immagini nelle 5 colonne
            st.image(blob_img, caption=f"Blob {i+1}", use_container_width=True, output_format="PNG")
