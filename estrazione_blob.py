def apply_colormap_final(image_gray):
    """
    Applica la colormap corretta ai blob estratti.
    Qui usiamo la colormap che ha dato il risultato desiderato.
    """
    final_cmap = plt.get_cmap("cividis")  # Puoi provare anche "turbo" o "jet" se necessario
    normalized_gray = image_gray / 255.0  # Normalizziamo i valori tra 0 e 1
    image_color = (final_cmap(normalized_gray)[:, :, :3] * 255).astype(np.uint8)
    return image_color

def extract_blobs(image, markers, maxima_map, original_size):
    """
    Estrae i blob segmentati e applica la colormap identificata correttamente.
    """
    img_np = np.array(image.convert("L"))  # Convertiamo in scala di grigi
    img_colored = apply_colormap_final(img_np)  # Applichiamo la colormap corretta
    
    blobs = []
    positions = []
    maxima_coords = []
    for label_id in np.unique(markers):
        if label_id <= 0:
            continue

        mask = (markers == label_id).astype(np.uint8) * 255
        x, y, w, h = cv2.boundingRect(mask)

        # Trova la posizione del massimo cromatico all'interno del blob
        subregion = maxima_map[y:y+h, x:x+w]
        _, _, _, max_loc = cv2.minMaxLoc(subregion)

        # Converti le coordinate del massimo nel sistema dell'immagine originale
        max_x_global = x + max_loc[0]
        max_y_global = y + max_loc[1]

        # Riporta le coordinate alla scala dell'immagine originale (pre-crop)
        scale_x = original_size[0] / img_np.shape[1]
        scale_y = original_size[1] / img_np.shape[0]
        max_x_original = int(max_x_global * scale_x)
        max_y_original = int(max_y_global * scale_y)

        # **Applicare la colormap corretta al blob ritagliato**
        cropped_blob = img_np[y:y+h, x:x+w]
        colored_blob = apply_colormap_final(cropped_blob)

        blob_pil = Image.fromarray(colored_blob)

        buf = io.BytesIO()
        blob_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()

        blobs.append((byte_im, (max_x_original, max_y_original)))  # Salva blob + coordinate globali
        positions.append((max_x_original, max_y_original))  # Posizione del massimo cromatico

    return blobs, positions
