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
    """Elabora l'immagine ritagliata per individuare i blob e visualizzarli."""
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

    blob_data = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cropped_blob = img_np[y:y+h, x:x+w]

        # Trova il punto con massima intensità nel blob
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img_gray[y:y+h, x:x+w])
        x_max, y_max = x + max_loc[0], y + max_loc[1]

        # Verifica se il blob ha un unico massimo o un gradiente monotono
        if is_monotonic_gradient(img_gray[y:y+h, x:x+w]):
            blob_pil = Image.fromarray(cropped_blob)
            buf = io.BytesIO()
            blob_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()

            blob_data.append((byte_im, x, x+w, y, y+h, x_max, y_max))
        else:
            # Se il gradiente non è monotono, suddividere in due parti
            half_h = h // 2
            sub_regions = [(y, y+half_h), (y+half_h, y+h)]
            for sub_y_start, sub_y_end in sub_regions:
                cropped_sub_blob = img_np[sub_y_start:sub_y_end, x:x+w]
                blob_pil = Image.fromarray(cropped_sub_blob)
                buf = io.BytesIO()
                blob_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()

                blob_data.append((byte_im, x, x+w, sub_y_start, sub_y_end, x_max, y_max))

    # Mostrare i blob ritagliati
    st.subheader("Blob Identificati")
    for i, (blob_img, x_start, x_end, y_start, y_end, x_max, y_max) in enumerate(blob_data):
        st.image(blob_img, caption=f"Blob {i+1}: X[{x_start}:{x_end}], Y[{y_start}:{y_end}], Max Intensità ({x_max}, {y_max})", use_container_width=True)

        # Pulsante di download per ogni blob
        st.download_button(
            label=f"📥 Scarica Blob {i+1}",
            data=blob_img,
            file_name=f"blob_{i+1}.png",
            mime="image/png"
        )
