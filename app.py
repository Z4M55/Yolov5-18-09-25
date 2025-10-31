# -*- coding: utf-8 -*-
import os
import cv2
import time
import sys
import torch
import numpy as np
import pandas as pd
import streamlit as st

# ===============================
# Configuración de página
# ===============================
st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🔎",
    layout="wide"
)

# ===============================
# Estilos (tema de colores + UI)
# ===============================
st.markdown("""
<style>
  :root {
    --bg: #B7E5CD;
    --fg: #305669;
    --fg-contrast: #ffffff;
    --fg-darker: #24454D;
  }
  html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: var(--bg) !important;
    color: var(--fg) !important;
  }
  [data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
    background-color: var(--bg) !important;
    color: var(--fg) !important;
    border-left: 1px solid rgba(48,86,105,.15);
  }
  h1, h2, h3, h4, h5, h6, p, label, span, div, .stMarkdown, .stCaption {
    color: var(--fg) !important;
  }
  .stButton > button {
    background-color: var(--fg) !important;
    color: var(--fg-contrast) !important;
    border: none !important;
    border-radius: 8px !important;
  }
  .stButton > button:hover { background-color: var(--fg-darker) !important; }
  .stDataFrame, .stDataFrame [class*="blank"], .stDataFrame [class*="row"] {
    color: var(--fg) !important;
  }
</style>
""", unsafe_allow_html=True)

# ===============================
# Carga del modelo (cacheada)
# ===============================
@st.cache_resource
def load_yolov5_model(model_name='yolov5s', pretrained=True, force_reload=False):
    """
    Carga YOLOv5 vía torch.hub de forma estable y lo envía a cuda/cpu.
    """
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=pretrained, force_reload=force_reload)
        model.to(device)
        return model, device
    except Exception as e:
        st.error(f"❌ No se pudo cargar YOLOv5 desde torch.hub: {e}")
        st.info("🔧 Sugerencia: `pip install torch torchvision torchaudio` (versión acorde a tu entorno)")
        return None, 'cpu'

# ===============================
# Título + descripción
# ===============================
st.title("🔎 Detección de Objetos en Imágenes")
st.markdown("""
Esta aplicación utiliza **YOLOv5** para detectar objetos en imágenes desde **cámara** o **archivo**.  
💡 **Tip:** Ajusta los parámetros en la barra lateral para mejorar precisión o velocidad.
""")

# ===============================
# Cargar el modelo
# ===============================
with st.spinner("🚀 Cargando modelo YOLOv5..."):
    model, device = load_yolov5_model()

if not model:
    st.stop()

# ===============================
# Barra lateral: parámetros
# ===============================
st.sidebar.title("🎛️ Parámetros")

conf = st.sidebar.slider('Confianza mínima', 0.0, 1.0, 0.25, 0.01)
iou = st.sidebar.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01)
imgsz = st.sidebar.select_slider('Tamaño de entrada (imgsz)', options=[320, 416, 480, 512, 640], value=640)

# Ajustes del modelo (si existen esos atributos)
if hasattr(model, "conf"):
    model.conf = conf
if hasattr(model, "iou"):
    model.iou = iou

st.sidebar.subheader('🧩 Filtrar clases (opcional)')
all_names = model.names if hasattr(model, "names") else {}
class_options = [f"{i}: {name}" for i, name in all_names.items()] if isinstance(all_names, dict) else []
chosen = st.sidebar.multiselect("Selecciona clases a detectar", class_options, default=[])
if chosen:
    model.classes = [int(opt.split(":")[0]) for opt in chosen]  # filtra en inferencia
else:
    model.classes = None

st.sidebar.caption(f"⚙️ Confianza: {conf:.2f} | IoU: {iou:.2f} | imgsz: {imgsz}")

# ===============================
# Entradas: cámara o archivo
# ===============================
col_cam, col_up = st.columns(2)
with col_cam:
    picture = st.camera_input("📷 Capturar imagen")
with col_up:
    upload = st.file_uploader("📁 Subir imagen", type=["jpg", "jpeg", "png"])

raw_bytes = None
if picture is not None:
    raw_bytes = picture.getvalue()
elif upload is not None:
    raw_bytes = upload.getvalue()

if raw_bytes is None:
    st.info("📸 Captura una imagen con la cámara o 📂 sube un archivo para continuar.")
    st.stop()

# ===============================
# Inferencia
# ===============================
start = time.time()

# Decodificar en BGR (OpenCV)
cv2_img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)

try:
    # Nota: en YOLOv5 clásico se puede pasar `size` para imgsz
    results = model(cv2_img, size=imgsz)
except Exception as e:
    st.error(f"❌ Error durante la detección: {e}")
    st.stop()

infer_sec = time.time() - start

# ===============================
# Parseo robusto de resultados
# ===============================
try:
    if hasattr(results, "pred"):         # versión clásica
        preds = results.pred[0]
    elif hasattr(results, "xyxy"):       # estructura alternativa
        preds = results.xyxy[0]
    else:
        raise AttributeError("Estructura de resultados inesperada.")
except Exception as e:
    st.error(f"❌ No se pudieron leer las predicciones: {e}")
    st.stop()

if preds is not None and len(preds):
    boxes = preds[:, :4]
    scores = preds[:, 4]
    cats   = preds[:, 5]
else:
    boxes, scores, cats = [], [], []

# ===============================
# Render: imagen anotada
# ===============================
try:
    results.render()  # anota internamente
    # results.ims o results.imgs según versión
    annotated = None
    if hasattr(results, "ims") and len(results.ims):
        annotated = results.ims[0]  # BGR
    elif hasattr(results, "imgs") and len(results.imgs):
        annotated = results.imgs[0]  # BGR
    if annotated is None:
        annotated = cv2_img
except Exception:
    annotated = cv2_img

# Convertir BGR->RGB para mostrar en Streamlit
annotated_rgb = annotated[:, :, ::-1]

# ===============================
# Layout de resultados
# ===============================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🖼️ Imagen con detecciones")
    st.image(annotated_rgb, use_container_width=True)

with col2:
    st.subheader("📦 Objetos detectados")
    if preds is not None and len(preds):
        label_names = model.names if hasattr(model, "names") else {}
        # Conteo por clase
        arr_cats = cats.cpu().numpy() if hasattr(cats, "cpu") else np.array(cats)
        unique_cats, counts = np.unique(arr_cats, return_counts=True)

        rows = []
        for cid, cnt in zip(unique_cats, counts):
            cid = int(cid)
            mask = (arr_cats == cid)
            # confianza promedio
            arr_scores = scores.cpu().numpy() if hasattr(scores, "cpu") else np.array(scores)
            conf_mean = float(arr_scores[mask].mean()) if mask.any() else 0.0
            rows.append({
                "Categoría": label_names.get(cid, str(cid)),
                "Cantidad": int(cnt),
                "Confianza promedio": f"{conf_mean:.2f}"
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        # Gráfico
        if not df.empty:
            st.bar_chart(df.set_index("Categoría")["Cantidad"])
    else:
        st.info("🙈 No se detectaron objetos con los parámetros actuales.")
        st.caption("📉 Prueba a reducir el umbral de confianza en la barra lateral.")

# ===============================
# Métricas + descarga
# ===============================
st.caption(f"⏱️ Tiempo de inferencia: {infer_sec*1000:.1f} ms | 💻 Device: {device} | 🖼️ imgsz: {imgsz}")

success, buf = cv2.imencode(".jpg", annotated)  # anotada en BGR
if success:
    st.download_button(
        "📥 Descargar imagen anotada",
        data=buf.tobytes(),
        file_name="detecciones.jpg",
        mime="image/jpeg"
    )

# ===============================
# Pie de página
# ===============================
st.markdown("---")
st.caption("🤖 App YOLOv5 + Streamlit — detección de objetos en tiempo real, con colores personalizados y UX con emojis.")
