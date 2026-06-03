import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import os

st.set_page_config(
    page_title="Flower Recognition",
    page_icon="",
    layout="centered"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stApp {
        background-color: #0a0a0a;
    }
    
    * {
        font-family: 'Courier New', 'SF Mono', monospace;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    
    .blinking-cursor {
        animation: blink 1s step-end infinite;
        display: inline-block;
        width: 10px;
    }
    
    .main-title {
        color: #c084fc;
        font-size: 1.8rem;
        margin-bottom: 1.5rem;
        font-weight: normal;
    }
    
    .stButton > button {
        background: transparent;
        color: #c084fc !important;
        border: 1px solid #c084fc !important;
        border-radius: 0px !important;
        font-family: 'Courier New', monospace !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        background: #c084fc20 !important;
    }
    
    .stProgress > div > div > div {
        background-color: #c084fc;
    }
    
    .stCaption {
        color: #8b5cf6;
    }
    
    hr {
        border-color: #c084fc30;
    }
    
    .result-box {
        border: 1px solid #c084fc;
        padding: 20px;
        margin-top: 20px;
        background-color: #0a0a0a;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">> flower recognition<span class="blinking-cursor">_</span></div>', unsafe_allow_html=True)

MODEL_FILE = "flowerpro.onnx"

@st.cache_resource
def load_model():
    return ort.InferenceSession(MODEL_FILE)

if not os.path.exists(MODEL_FILE):
    st.error("> flowerpro.onnx not found")
    st.stop()

try:
    session = load_model()
    st.success("> model loaded")
except Exception as e:
    st.error(f"> error: {e}")
    st.stop()

input_info = session.get_inputs()[0]
input_shape = input_info.shape
target_size = (input_shape[1], input_shape[2])
num_classes = session.get_outputs()[0].shape[1]

CLASS_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

st.markdown("""
> nhan dien 5 loai hoa pho bien
> buoc 1: chon file anh (jpg, jpeg, png)
> buoc 2: nhan nut predict
> buoc 3: xem ket qua va do tin cay
""")

uploaded = st.file_uploader("", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

if uploaded:
    image = Image.open(uploaded)
    st.image(image, width=250)
    
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    image = image.resize(target_size)
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    if st.button("> predict"):
        input_name = input_info.name
        predictions = session.run(None, {input_name: img_array})[0][0]
        
        idx = np.argmax(predictions)
        confidence = float(predictions[idx])
        
        st.markdown(f"### > {CLASS_NAMES[idx]}")
        st.caption(f"confidence: {confidence:.2%}")
        
        st.markdown("---")
        st.markdown("> top 5")
        top5_idx = np.argsort(predictions)[-5:][::-1]
        for i, idx in enumerate(top5_idx, 1):
            prob = float(predictions[idx])
            st.progress(prob, text=f"{i}. {CLASS_NAMES[idx]} - {prob:.2%}")

st.markdown("---")
st.caption("> version 1.0 | flower recognition cnn")
