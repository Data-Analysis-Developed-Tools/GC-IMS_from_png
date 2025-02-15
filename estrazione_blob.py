import cv2
import numpy as np
import io
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from scipy.ndimage import label
import matplotlib.pyplot as plt

def apply_colormap_parula(image_gray):
    """
    Applica la mappa cromatica 'Parula' all'immagine in scala di grigi.
    """
    parula_cmap = plt.get_cmap("cividis")  # "Cividis" è molto simile a "Parula"
    normalized_gray = image_gray / 255.0  # Normalizziamo tra 0 e 1
    image_color = (parula_cmap(normalized_gray)[:, :, :3] * 255).astype(np.uint8)
    return image_color

def find_maxima(image_gray, threshold_factor, neighborhood_size):
    """
    Identifica i punti di massimo cromatico con maggiore sensibilità.
    """
    blurred = cv2.GaussianBlur(image_gray, (3, 3), 0)
    min_val, max_val, _, _ = cv2.minMaxLoc(blurred)
    threshold = max_val * threshold_factor
    maxima_mask = (blurred >= threshold).astype(np.uint8) * 255

    kernel = np.ones((neighborhood_size, neighborhood_size), np.uint8)
    local_maxima = cv2.dilate(maxima_mask, kernel, iterations=1)
    
    labeled_maxima, num_features = label(local_maxima)

    return labeled_maxima, num_features

def apply_watershed(image_gray, opening_iter):
    """
    Applica Watershed per segmentare i blob.
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
    Estrae i blob segmentati e applica la colormap 'Parula' per uniformità.
    """
    img_np = np.array(image.convert("L"))  # Convertiamo in scala di grigi
    img_colored = apply_colormap_parula(img_np)  # Applichiamo la colormap Parula
    
    blobs = []
    positions = []
    for label_id in np.unique(markers):
        if label_id <= 0:
            continue

        mask = (markers == label_id).astype(np.uint8) * 255
        x, y, w, h = cv2.boundingRect(mask)

        subregion = maxima_map[y:y+h, x:x+w]
        num_maxima = len(np.unique(subregion)) - 1

        if num_maxima == 1:
            cropped_blob = img_colored[y:y+h, x:x+w]

            blob_pil = Image.fromarray(cropped_blob)

            buf = io.BytesIO()
            blob_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()

            blobs.append(byte_im)
            positions.append((x + w // 2, y + h // 2))  # Centro del blob per la numerazione

    return blobs, positions

def draw_blob_numbers(image, positions):
    """
    Disegna il numero di ogni blob sopra l'immagine ritagliata con dimensioni più piccole.
    """
    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)

    font_size = 10  # **Numero ridotto a 1/3 delle dimensioni precedenti**
    
    for i, (x, y) in enumerate(positions):
        draw.text((x, y), str(i + 1), fill="white", stroke_fill="black", stroke_width=1, anchor="mm")

    return image_pil

def process_image(image):
    """Segmenta l'immagine e applica la mappa cromatica 'Parula' ai blob estratti."""
    img_np = np.array(image.convert("L"))  # Convertiamo in scala di grigi
    img_colored = apply_colormap_parula(img_np)   # Applichiamo la colormap "Parula"

    # 🔹 Cursore per la sensibilità dei massimi cromatici
    threshold_factor = st.sidebar.slider("Sensibilità ai massimi cromatici", 0.7, 1.0, 0.90, step=0.01)

    # 🔹 Cursore per la dimensione del filtro per la ricerca dei massimi
    neighborhood_size = st.sidebar.slider("Dimensione del filtro per massimi", 1, 7, 3, step=2)

    # 🔹 Cursore per la separazione dei blob
    opening_iter = st.sidebar.slider("Separazione blob (iterazioni apertura)", 1, 5, 2)

    # 1️⃣ Identifica i massimi con parametri regolabili
    maxima_map, _ = find_maxima(img_np, threshold_factor, neighborhood_size)

    # 2️⃣ Applica Watershed per segmentare i blob
    markers = apply_watershed(img_np, opening_iter)

    # 3️⃣ Estrae i blob e memorizza le loro posizioni per numerazione
    blob_images, positions = extract_blobs(image, markers, maxima_map)

    # 4️⃣ Disegna i numeri sui blob dell'immagine ritagliata
    img_annotated = draw_blob_numbers(img_colored, positions)

    # Mostrare l'immagine segmentata con la mappa cromatica e numeri
    st.subheader("Immagine Segmentata con Parula e Numerazione Blob")
    st.image(img_annotated, use_container_width=True, caption="Immagine con Blob Numerati")

    # Mostrare i blob in una griglia a 5 colonne
    st.subheader("Galleria di Blob Identificati")
    cols = st.columns(5)

    for i, blob_img in enumerate(blob_images):
        with cols[i % 5]:
            st.image(blob_img, caption=f"Blob {i+1}", use_container_width=True, output_format="PNG")
