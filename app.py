# -*- coding: utf-8 -*-
import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

# =========================
# Configuración de la página
# =========================
st.set_page_config(
    page_title="🤖 Detección de Objetos | Tech Mode",
    page_icon="🧠",
    layout="wide"
)

# =========================
# Estilos Tech (oscuro + neón)
# =========================
st.markdown("""
<style>
  :root{
    --bg:#0b1220;         /* fondo principal */
    --panel:#0f182b;      /* paneles/cards */
    --text:#e6f7ff;       /* texto principal */
    --muted:#9fb3c8;      /* texto secundario */
    --accent:#00e5ff;     /* cian neón */
    --accent2:#00ffa3;    /* verde neón */
    --danger:#ff4d4f;
  }
  html, body, .stApp, [data-testid="stAppViewContainer"]{
    background: radial-gradient(1000px 600px at 10% 0%, #0f1a30 0%, var(--bg) 60%);
    color: var(--text) !important;
  }
  [data-testid="stSidebar"], section[data-testid="stSidebar"] > div{
    background: linear-gradient(180deg, #0e1628 0%, #0b1220 100%) !important;
    color: var(--text) !important;
    border-right: 1px solid rgba(0,229,255,.15);
  }
  .block-container{
    padding-top: 1.2rem;
  }
  h1,h2,h3,h4,h5,h6{
    color: var(--text) !important;
    font-family: "JetBrains Mono", Consolas, Menlo, monospace;
    letter-spacing: .5px;
  }
  p, label, span, div, .stMarkdown{
    color: var(--text) !important;
    font-family: "Inter", system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  }
  .stCaption, .st-emotion-cache-12fmjuu, .st-emotion-cache-1lb3x6j{
    color: var(--muted) !important;
  }
  .stButton>button{
    background: linear-gradient(90deg, var(--accent) 0%, var(--accent2) 100%) !important;
    color: #00121a !important;
    border: 0 !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    box-shadow: 0 0 12px rgba(0,229,255,.5), inset 0 0 0 rgba(0,0,0,0);
    transition: transform .08s ease-in-out, box-shadow .2s ease-in-out;
  }
  .stButton>button:hover{ transform: translateY(-1px); box-shadow: 0 0 16px rgba(0,229,255,.75); }
  .stSlider label, .stNumberInput label, .stSelectbox label, .stFileUploader label{
    color: var(--muted) !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .6px;
    font-size: .85rem;
  }
  /* cuadros/expander */
  .stApp [data-testid="stExpander"]{
    background: var(--panel) !important;
    border: 1px solid rgba(0,229,255,.15);
    border-radius: 12px;
  }
  /* DataFrame */
  .stDataFrame div, .stDataFrame table{
    color: var(--text) !important;
  }
  /* Inputs */
  .stTextInput>div>div>input, .stNumberInput input, .stFileUploader, .stSelectbox div[data-baseweb="select"]{
    background: #0f182b !important;
    color: var(--text) !important;
    border: 1px solid rgba(0,229,255,.2) !important;
    border-radius: 10px !important;
  }
</style>
""", unsafe_allow_html=True)

# =========================
# Carga del modelo YOLOv5
# =========================
@st.cache_resource
def load_yolov5_model(model_path='yolov5s.pt'):
    """
    Intenta cargar con 'yolov5.load'; si falla, usa torch.hub (pretrained).
    """
    try:
        import yolov5
        try:
            model = yolov5.load(model_path, weights_only=False)
            return model
        except TypeError:
            model = yolov5.load(model_path)
            return model
    except Exception as e:
        st.warning(f"⚠️ Fallback torch.hub: {e}")
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            model.to(device)
            return model
        except Exception as e2:
            st.error(f"❌ Error alternativo: {e2}")
            return None

# =========================
# Título principal
# =========================
st.title("🛰️ Detección de Objetos | Tech Mode")
st.markdown("""
Sistema de visión por computador **YOLOv5** para analizar imágenes desde **cámara** o **archivo**.  
Ajusta parámetros para equilibrar **precisión** y **velocidad**.  
**Stack:** PyTorch · OpenCV · Streamlit ⚙️
""")

# =========================
# Carga del modelo
# =========================
with st.spinner("🔧 Inicializando modelo… Cargando pesos y optimizaciones…"):
    model = load_yolov5_model()

if model is None:
    st.error("💥 No se pudo cargar el modelo. Revisa dependencias e inténtalo otra vez.")
    st.stop()

# =========================
# Sidebar: parámetros
# =========================
st.sidebar.title("🎛️ Panel de Control")
model.conf = st.sidebar.slider('Confianza mínima (conf) 🔎', 0.0, 1.0, 0.25, 0.01)
model.iou  = st.sidebar.slider('Umbral IoU (NMS) 🧩', 0.0, 1.0, 0.45, 0.01)

with st.sidebar.expander("⚙️ Opciones avanzadas"):
    try:
        model.agnostic   = st.checkbox('NMS sin clase (agnostic)', False)
        model.multi_label= st.checkbox('Múltiples etiquetas por caja', False)
        model.max_det    = st.number_input('Máx. detecciones', 10, 2000, 1000, 10)
    except:
        st.info("🔬 Algunas opciones no están disponibles en esta build.")

# =========================
# Captura/carga de imagen
# =========================
st.markdown("### 📸 Fuente de imagen")
col1, col2 = st.columns([2,1])
with col1:
    picture = st.camera_input("Cam 🎥 (tomar foto)")
with col2:
    uploaded = st.file_uploader("Archivo 📂 (JPG/PNG)", type=["jpg", "jpeg", "png"])

if picture or uploaded:
    image_source = picture if picture else uploaded
    bytes_data = image_source.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # =========================
    # Detección
    # =========================
    with st.spinner("🧠 Inferencia en curso…"):
        try:
            results = model(cv2_img)
            results.render()  # anota internamente
        except Exception as e:
            st.error(f"❌ Error durante la detección: {e}")
            st.stop()

    # Recuperar imagen anotada real (no el frame original)
    annotated = None
    if hasattr(results, "ims") and len(results.ims) > 0:
        annotated = results.ims[0]  # BGR
    elif hasattr(results, "imgs") and len(results.imgs) > 0:
        annotated = results.imgs[0]  # BGR
    if annotated is None:
        annotated = cv2_img

    annotated_rgb = annotated[:, :, ::-1]

    st.markdown("## 🧾 Resultados")
    col_img, col_data = st.columns(2)

    with col_img:
        st.subheader("🖼️ Imagen anotada")
        st.image(annotated_rgb, use_container_width=True, caption="Cajas y etiquetas por YOLOv5")

    with col_data:
        # Parsing robusto
        try:
            predictions = results.pred[0]
        except Exception:
            predictions = None

        if predictions is not None and len(predictions) > 0:
            boxes = predictions[:, :4]
            scores = predictions[:, 4]
            cats   = predictions[:, 5]
            labels = model.names

            # Conteo por clase
            category_count = {}
            for c in cats:
                idx = int(c.item()) if hasattr(c, "item") else int(c)
                category_count[idx] = category_count.get(idx, 0) + 1

            # Tabla de resumen
            data = []
            for idx, count in category_count.items():
                label = labels[idx] if isinstance(labels, dict) else str(idx)
                # confianza promedio por clase
                mask = (cats == idx) if not hasattr(cats, "cpu") else (cats.cpu().numpy() == idx)
                conf_mean = float(scores[mask].mean().item() if hasattr(scores[mask].mean(), "item") else scores[mask].mean())
                data.append({
                    "🔖 Clase": label,
                    "🔢 Cantidad": count,
                    "📈 Conf. Promedio": f"{conf_mean:.2f}"
                })

            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
            if not df.empty:
                st.bar_chart(df.set_index("🔖 Clase")["🔢 Cantidad"])
        else:
            st.info("🙈 Sin detecciones con los parámetros actuales. Prueba a bajar el umbral de confianza.")

else:
    st.info("💡 Usa la **cámara** o sube un **archivo** para comenzar la detección.")

# =========================
# Pie de página
# =========================
st.markdown("---")
st.markdown("""
**Stack:** PyTorch · YOLOv5 · OpenCV · Streamlit  
**Tema:** _Tech masculine_ — fondo oscuro + neón cian/verde.  
**Tip:** Ajusta `conf`/`IoU` para equilibrar precisión y velocidad. ⚙️
""")
st.caption("© 2025 • Vision AI • Modo Tecnológico 🧠")
