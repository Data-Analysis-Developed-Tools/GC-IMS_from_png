import cv2
import numpy as np
import io
from PIL import Image
import streamlit as st
from scipy.ndimage import label

def find_maxima(image_gray, threshold_factor, neighborhood_size):
    """
    Identifica i punti di massimo cromatico con maggiore sensibilità.
    - threshold_factor: sensibilità alla selezione dei massimi.
    - neighborhood_size: ampiezza del filtro per la ricerca dei massimi.
    """
    # Sfoca leggermente l'immagine per eliminare il rumore
    blurred = cv2.GaussianBlur(image_gray, (3, 3), 0)

    # Trova il valore massimo globale
    min_val, max_val, _, _ = cv2.minMaxLoc(blurred)

    # Soglia regolabile per individuare i massimi
    threshold = max_val * threshold_factor

    # Creazione di una maschera dei massimi
    maxima_mask = (blurred >= threshold).astype(np.uint8) * 255

    # Usa dilatazione per enfatizzare i picchi locali
    kernel = np.ones((neighborhood_size, neighborhood_size), np.uint8)
    local_maxima = cv2.dilate(maxima_mask, kernel, iterations=1)

    # Etichettatura delle aree massime
    labeled_maxima, num_features = label(local_maxima)

    return labeled_maxima, num_features

def apply_watershed(image_gray, opening_iter):
    """
    Applica Watershed con parametri regolabili.
    """
    ret, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=opening_iter)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR), markers)

    return markers

def extract_blobs(image, markers, maxima_map):
    """
    Estrae i blob segmentati e garantisce che ciascuno abbia un solo massimo cromatico.
    """
    img_np = np.array(image.convert("RGB"))
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    blobs = []
    for label_id in np.unique(markers):
        if label_id <= 0:  # Ignora il background
            continue

        mask = (markers == label_id).astype(np.uint8) * 255
        x, y, w, h = cv2.boundingRect(mask)

        subregion = maxima_map[y:y+h, x:x+w]
        num_maxima = len(np.unique(subregion)) - 1  

        if num_maxima == 1:
            cropped_blob = img_np[y:y+h, x:x+w]

            cropped_blob = cv2.cvtColor(cropped_blob, cv2.COLOR_BGR2RGB)

            blob_pil = Image.fromarray(cropped_blob, mode="RGB")

            buf = io.BytesIO()
            blob_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()

            blobs.append(byte_im)

    return blobs

def process_image(image):
    """Segmenta l'immagine con Watershed e garantisce un massimo cromatico per blob."""
    img_np = np.array(image.convert("RGB"))
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # 🔹 Cursore per la soglia dei massimi cromatici
    threshold_factor = st.sidebar.slider("Sensibilità ai massimi cromatici", 0.7, 1.0, 0.90, step=0.01)

    # 🔹 Cursore per la dimensione del filtro di ricerca dei massimi
    neighborhood_size = st.sidebar.slider("Dimensione del filtro per massimi", 1, 7, 3, step=2)

    # 🔹 Cursore per iterazioni di apertura morfologica
    opening_iter = st.sidebar.slider("Separazione blob (iterazioni apertura)", 1, 5, 2)

    # 1️⃣ Identifica i massimi con parametri regolabili
    maxima_map, _ = find_maxima(img_gray, threshold_factor, neighborhood_size)

    # 2️⃣ Applica Watershed con parametri regolabili
    markers = apply_watershed(img_gray, opening_iter)

    # 3️⃣ Estrae i blob assicurandosi che abbiano un solo massimo
    blob_images = extract_blobs(image, markers, maxima_map)

    # Mostrare i blob ritagliati in una griglia a 5 colonne
    st.subheader("Galleria di Blob Identificati")
    cols = st.columns(5)

    for i, blob_img in enumerate(blob_images):
        with cols[i % 5]:
            st.image(blob_img, caption=f"Blob {i+1}", use_container_width=True, output_format="PNG")
