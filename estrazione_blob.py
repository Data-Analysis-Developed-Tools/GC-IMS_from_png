import cv2
import numpy as np
import io
from PIL import Image
import streamlit as st
from scipy.ndimage import label

def find_maxima(image_gray):
    """
    Identifica i punti di massimo cromatico su tutta l'immagine.
    Restituisce una maschera con i massimi.
    """
    min_val, max_val, _, _ = cv2.minMaxLoc(image_gray)

    # Soglia per trovare i massimi: consideriamo i pixel molto vicini al massimo globale
    threshold = max_val * 0.95  # Adatta la soglia per evitare noise
    maxima_mask = (image_gray >= threshold).astype(np.uint8) * 255

    # Etichettare i massimi
    labeled_maxima, num_features = label(maxima_mask)

    return labeled_maxima, num_features

def apply_watershed(image_gray):
    """
    Utilizza Watershed per segmentare i blob prima di estrarli.
    """
    # Converti in formato adatto per Watershed
    ret, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Rimuove il rumore
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Trova lo sfondo certo
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Trova il primo piano certo
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

    # Trova regioni incerte
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Etichette dei marker
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Applica Watershed
    markers = cv2.watershed(cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR), markers)

    return markers

def extract_blobs(image, markers, maxima_map):
    """
    Estrae i blob segmentati dall'immagine usando Watershed.
    Garantisce che ogni blob abbia un massimo univoco.
    """
    img_np = np.array(image.convert("RGB"))
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    blobs = []
    for label_id in np.unique(markers):
        if label_id <= 0:  # Ignora il background
            continue

        # Trova il bounding box del blob
        mask = (markers == label_id).astype(np.uint8) * 255
        x, y, w, h = cv2.boundingRect(mask)

        # Controlla se il blob ha un solo massimo nella mappa
        subregion = maxima_map[y:y+h, x:x+w]
        num_maxima = len(np.unique(subregion)) - 1  # Rimuove lo sfondo

        if num_maxima == 1:
            cropped_blob = img_np[y:y+h, x:x+w]

            # Assicura che sia un'immagine RGB compatibile con PIL
            cropped_blob = cv2.cvtColor(cropped_blob, cv2.COLOR_BGR2RGB)

            # Converte in immagine PIL
            blob_pil = Image.fromarray(cropped_blob, mode="RGB")

            # Converte PIL Image in PNG Bytes
            buf = io.BytesIO()
            blob_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()

            blobs.append(byte_im)

    return blobs

def process_image(image):
    """Segmenta l'immagine usando Watershed e garantisce un massimo per blob."""
    img_np = np.array(image.convert("RGB"))
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # 1️⃣ Identifica i massimi su tutta l'immagine
    maxima_map, _ = find_maxima(img_gray)

    # 2️⃣ Applica Watershed per segmentare i blob
    markers = apply_watershed(img_gray)

    # 3️⃣ Estrae i blob assicurandosi che abbiano un solo massimo
    blob_images = extract_blobs(image, markers, maxima_map)

    # Mostrare i blob ritagliati in una griglia a 5 colonne
    st.subheader("Galleria di Blob Identificati")
    cols = st.columns(5)  # Creiamo 5 colonne

    for i, blob_img in enumerate(blob_images):
        with cols[i % 5]:  # Distribuisce le immagini nelle 5 colonne
            st.image(blob_img, caption=f"Blob {i+1}", use_container_width=True, output_format="PNG")
