# --- CARGA DE MODELO (simple, cacheada y con device) ---
@st.cache_resource
def load_yolov5_model(model_name='yolov5s', pretrained=True, force_reload=False):
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=pretrained, force_reload=force_reload)
        model.to(device)
        return model, device
    except Exception as e:
        st.error(f"❌ No se pudo cargar YOLOv5 desde torch.hub: {e}")
        st.info("Prueba: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 (o cpu)")
        return None, 'cpu'

with st.spinner("Cargando modelo YOLOv5..."):
    model, device = load_yolov5_model()

if not model:
    st.stop()

# --- SIDEBAR: parámetros cómodos ---
st.sidebar.title("Parámetros")
conf = st.sidebar.slider('Confianza mínima', 0.0, 1.0, 0.25, 0.01)
iou = st.sidebar.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01)
imgsz = st.sidebar.select_slider('Tamaño de entrada (imgsz)', options=[320, 416, 480, 512, 640], value=640)

st.sidebar.subheader('Clases')
all_names = model.names if hasattr(model, "names") else {}
class_options = [f"{i}: {name}" for i, name in all_names.items()] if isinstance(all_names, dict) else []
chosen = st.sidebar.multiselect("Filtrar clases (opcional)", class_options, default=[])
if chosen:
    model.classes = [int(opt.split(":")[0]) for opt in chosen]  # filtra en inferencia
else:
    model.classes = None

# Ajustes del modelo
model.conf = conf
model.iou = iou

# --- ENTRADAS: cámara o archivo ---
col_cam, col_up = st.columns(2)
with col_cam:
    picture = st.camera_input("📷 Capturar imagen")
with col_up:
    upload = st.file_uploader("📁 Subir imagen", type=["jpg","jpeg","png"])

# Selecciona la fuente de imagen
raw_bytes = None
if picture is not None:
    raw_bytes = picture.getvalue()
elif upload is not None:
    raw_bytes = upload.getvalue()

if raw_bytes is None:
    st.info("Captura una imagen con la cámara o sube un archivo para continuar.")
    st.stop()

# --- INFERENCIA ---
import time
start = time.time()

# decodifica en BGR (cv2) o usa PIL; aquí usamos cv2 para ser coherentes con tu flujo
cv2_img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)

try:
    # Nota: puedes pasar imgsz como kwarg a __call__
    results = model(cv2_img, size=imgsz)
except Exception as e:
    st.error(f"❌ Error durante la detección: {e}")
    st.stop()

infer_sec = time.time() - start

# --- PARSEO ROBUSTO (pred o xyxy) ---
try:
    if hasattr(results, "pred"):         # versiones clásicas
        preds = results.pred[0]
    elif hasattr(results, "xyxy"):       # alternativa
        preds = results.xyxy[0]
    else:
        raise AttributeError("Estructura de resultados inesperada.")
except Exception as e:
    st.error(f"❌ No se pudieron leer las predicciones: {e}")
    st.stop()

boxes = preds[:, :4] if preds is not None and len(preds) else []
scores = preds[:, 4] if preds is not None and len(preds) else []
cats   = preds[:, 5] if preds is not None and len(preds) else []

# --- VISUALIZACIÓN CORRECTA CON CAJAS ---
# results.render() modifica imagenes internas; hay que mostrarlas, no el cv2_img original
try:
    results.render()  # anota internamente
    # según versión puede ser results.ims o results.imgs; manejamos ambos
    annotated = None
    if hasattr(results, "ims") and len(results.ims):
        annotated = results.ims[0]  # BGR
    elif hasattr(results, "imgs") and len(results.imgs):
        annotated = results.imgs[0]  # BGR
    if annotated is None:
        # fallback: muestra original si algo cambia
        annotated = cv2_img
except Exception:
    annotated = cv2_img

# Convertir BGR->RGB para Streamlit
annotated_rgb = annotated[:, :, ::-1]

col1, col2 = st.columns(2)
with col1:
    st.subheader("Imagen con detecciones")
    st.image(annotated_rgb, use_container_width=True)

with col2:
    st.subheader("Objetos detectados")
    if preds is not None and len(preds):
        # conteo por clase + confianza promedio
        label_names = model.names if hasattr(model, "names") else {}
        # contadores por clase
        unique_cats, counts = np.unique(cats.cpu().numpy() if hasattr(cats, "cpu") else np.array(cats), return_counts=True)
        rows = []
        for cid, cnt in zip(unique_cats, counts):
            cid = int(cid)
            mask = (cats == cid)
            conf_mean = float(scores[mask].mean().item() if hasattr(scores[mask].mean(), "item") else scores[mask].mean())
            rows.append({"Categoría": label_names.get(cid, str(cid)),
                         "Cantidad": int(cnt),
                         "Confianza promedio": f"{conf_mean:.2f}"})
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df.set_index("Categoría")["Cantidad"])
    else:
        st.info("No se detectaron objetos con los parámetros actuales.")
        st.caption("Prueba a reducir la confianza en la barra lateral.")

# --- MÉTRICAS + DESCARGA ---
st.caption(f"⏱️ Tiempo de inferencia: {infer_sec*1000:.1f} ms | Device: {device} | imgsz: {imgsz}")
# descarga de la imagen anotada
success, buf = cv2.imencode(".jpg", annotated)  # anotada aún en BGR; ok para JPG
if success:
    st.download_button("📥 Descargar imagen anotada", data=buf.tobytes(), file_name="detecciones.jpg", mime="image/jpeg")
