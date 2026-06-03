import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

st.set_page_config(
    page_title="Flower Recognition",
    page_icon="",
    layout="wide"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stApp {
        background-color: #ffffff;
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
    
    h1 {
        color: #000000;
        font-weight: normal;
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    
    .stButton > button {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 0px !important;
        font-family: 'Courier New', monospace !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        background-color: #333333 !important;
    }
    
    .result-box {
        border: 1px solid #000000;
        padding: 20px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1>> flower recognition<span class="blinking-cursor">_</span></h1>', unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="flowerpro.tflite")
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    target_size = (input_details[0]['shape'][1], input_details[0]['shape'][2])
    st.success("> model loaded")
except:
    st.error("> model not found")
    st.stop()

CLASS_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

st.markdown("""
> nhan dien 5 loai hoa pho bien
> buoc 1: chon file anh (jpg, jpeg, png)
> buoc 2: nhan nut predict
> buoc 3: xem ket qua
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
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
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
