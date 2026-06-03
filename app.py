import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
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
    
    .main-title {
        color: #ff69b4;
        font-size: 2rem;
        margin-bottom: 1rem;
        font-weight: normal;
    }
    
    .stButton > button {
        background: transparent;
        color: #ff69b4 !important;
        border: 1px solid #ff69b4 !important;
        border-radius: 0px !important;
        font-family: 'Courier New', monospace !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        background: #ff69b420 !important;
    }
    
    .stProgress > div > div > div {
        background-color: #ff69b4;
    }
    
    .stCaption {
        color: #ff69b4;
    }
    
    hr {
        border-color: #ff69b450;
    }
    
    .result-box {
        border: 1px solid #ff69b4;
        padding: 20px;
        margin-top: 20px;
        background-color: #ffffff;
    }
    
    .flower-card {
        border: 1px solid #ff69b4;
        padding: 15px;
        margin-bottom: 10px;
        background-color: #fffafc;
    }
    
    .flower-name {
        color: #ff69b4;
        font-size: 1rem;
        font-weight: bold;
        margin-bottom: 5px;
    }
    
    .flower-desc {
        color: #333333;
        font-size: 0.7rem;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">> flower recognition<span class="blinking-cursor">_</span></div>', unsafe_allow_html=True)

MODEL_FILE = "flowerpro.onnx"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        return ort.InferenceSession(MODEL_FILE)
    return None

session = load_model()

if session is None:
    st.error("> flowerpro.onnx not found")
    st.stop()

input_info = session.get_inputs()[0]
input_shape = input_info.shape
target_size = (input_shape[1], input_shape[2])

# Thông tin chi tiết về các loài hoa
FLOWER_INFO = {
    'daisy': {
        'name': 'Hoa Cúc',
        'color': 'Trắng, vàng',
        'meaning': 'Tượng trưng cho sự ngây thơ, trong sáng và tình yêu thuần khiết',
        'season': 'Mùa xuân - hạ'
    },
    'dandelion': {
        'name': 'Hoa Bồ Công Anh',
        'color': 'Vàng',
        'meaning': 'Biểu tượng của sự lạc quan, hy vọng và sức mạnh tinh thần',
        'season': 'Mùa xuân - thu'
    },
    'rose': {
        'name': 'Hoa Hồng',
        'color': 'Đỏ, hồng, trắng, vàng',
        'meaning': 'Tình yêu, đam mê và sự lãng mạn',
        'season': 'Quanh năm'
    },
    'sunflower': {
        'name': 'Hoa Hướng Dương',
        'color': 'Vàng',
        'meaning': 'Sự trung thành, may mắn và hạnh phúc',
        'season': 'Mùa hè'
    },
    'tulip': {
        'name': 'Hoa Tulip',
        'color': 'Đỏ, hồng, vàng, tím',
        'meaning': 'Tình yêu hoàn hảo, sang trọng và quý phái',
        'season': 'Mùa xuân'
    }
}

CLASS_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Layout 2 cột
col_left, col_right = st.columns([0.5, 0.5])

with col_left:
    st.markdown("### > upload")
    uploaded = st.file_uploader("", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, width=280)
        
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
            flower_key = CLASS_NAMES[idx]
            flower = FLOWER_INFO[flower_key]
            
            st.markdown("---")
            st.markdown(f"### > {flower['name']}")
            st.caption(f"confidence: {confidence:.2%}")
            
            st.markdown("---")
            st.markdown("> thong tin hoa")
            st.write(f"**mau sac:** {flower['color']}")
            st.write(f"**y nghia:** {flower['meaning']}")
            st.write(f"**mua:** {flower['season']}")
            
            st.markdown("---")
            st.markdown("> top 5")
            top5_idx = np.argsort(predictions)[-5:][::-1]
            for i, idx in enumerate(top5_idx, 1):
                prob = float(predictions[idx])
                name = FLOWER_INFO[CLASS_NAMES[idx]]['name']
                st.progress(prob, text=f"{i}. {name} - {prob:.2%}")

with col_right:
    st.markdown("### > flower library")
    st.caption("5 loai hoa pho bien")
    
    for flower_key in CLASS_NAMES:
        flower = FLOWER_INFO[flower_key]
        with st.expander(f"🌸 {flower['name']}"):
            st.markdown(f"""
            <div class="flower-card">
                <div class="flower-name">{flower['name']}</div>
                <div class="flower-desc"><b>mau sac:</b> {flower['color']}</div>
                <div class="flower-desc"><b>y nghia:</b> {flower['meaning']}</div>
                <div class="flower-desc"><b>mua:</b> {flower['season']}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.caption("> version 1.0 | flower recognition cnn")
