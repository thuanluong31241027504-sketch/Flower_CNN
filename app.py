import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import os

st.set_page_config(page_title="Flower Recognition", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {background-color: #ffffff;}
    * {font-family: 'Courier New', monospace;}
    h1 {color: #000000; font-weight: normal;}
    .stButton > button {background-color: #000000 !important; color: #ffffff !important; border-radius: 0px !important; width: 100% !important;}
</style>
""", unsafe_allow_html=True)

st.markdown("# > flower recognition_")

MODEL_FILE = "flowerpro.onnx"

# Kiểm tra file tồn tại
if not os.path.exists(MODEL_FILE):
    st.error(f"❌ Không tìm thấy file: {MODEL_FILE}")
    st.write("Các file trong thư mục:")
    for f in os.listdir('.'):
        st.write(f"  - {f}")
    st.stop()

# Load model
@st.cache_resource
def load_model():
    return ort.InferenceSession(MODEL_FILE)

try:
    session = load_model()
    st.success("✅ Model loaded")
except Exception as e:
    st.error(f"Lỗi load model: {e}")
    st.stop()

input_info = session.get_inputs()[0]
target_size = (input_info.shape[1], input_info.shape[2])
CLASS_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

st.markdown("""
> buoc 1: chon file anh
> buoc 2: nhan nut predict
""")

uploaded = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, width=250)
    
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    image = image.resize(target_size)
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    if st.button("> predict"):
        pred = session.run(None, {input_info.name: img_array})[0]
        idx = np.argmax(pred[0])
        conf = float(pred[0][idx])
        
        st.success(f"### {CLASS_NAMES[idx]}")
        st.info(f"confidence: {conf:.2%}")
        
        # Top 3
        st.markdown("> top 3")
        top3 = np.argsort(pred[0])[-3:][::-1]
        for i, idx in enumerate(top3, 1):
            st.progress(float(pred[0][idx]), text=f"{i}. {CLASS_NAMES[idx]} - {pred[0][idx]:.2%}")
