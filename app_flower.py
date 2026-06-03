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

# Custom CSS (giống app món ăn)
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stApp {
        background-color: #ffffff;
    }
    
    html, body, [class*="css"] {
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
        font-family: 'Courier New', monospace;
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }
    
    .stFileUploader > div {
        background-color: #f5f5f5;
        border: 1px solid #000000;
        border-radius: 0px;
    }
    
    .stButton > button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #000000 !important;
        border-radius: 0px !important;
        padding: 0.5rem 1rem !important;
        font-family: 'Courier New', monospace !important;
        font-weight: bold !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    .stProgress > div > div > div {
        background-color: #000000;
    }
    
    .stCaption {
        color: #666666;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<h1>
    > Flower Recognition<span class="blinking-cursor">_</span>
</h1>
""", unsafe_allow_html=True)

# Model path
MODEL_PATH = "flower.onnx"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return ort.InferenceSession(MODEL_PATH)
    return None

session = load_model()

if session is None:
    st.error("> flower.onnx not found")
    st.stop()

# Lấy input shape từ model
input_info = session.get_inputs()[0]
input_shape = input_info.shape
target_size = (input_shape[1], input_shape[2])

# App introduction
st.markdown("""
> nhan dien cac loai hoa pho bien
> su dung mo hinh CNN voi do chinh xac cao
> tai anh bat ky de nhan dien ngay
""")

# Instruction
st.markdown("""
> buoc 1: chon file anh (jpg, jpeg, png)
> buoc 2: nhan nut predict
> buoc 3: xem ket qua va do tin cay
""")

# Flower data (THAY THEO ĐÚNG CLASS CỦA MODEL BẠN)
# Bạn cần xem model flower.h5 có bao nhiêu class và tên gì
FLOWER_DATA = {
    'Hoa Hong': 'Hoa hồng - Biểu tượng của tình yêu, cánh mềm mại, nhiều màu sắc',
    'Hoa Cuc': 'Hoa cúc - Tượng trưng cho sự trường thọ, cánh vàng rực rỡ',
    'Hoa Lan': 'Hoa lan - Thanh cao, quý phái, có nhiều loại đẹp',
    'Hoa Sen': 'Hoa sen - Quốc hoa Việt Nam, tượng trưng cho sự thanh cao',
    'Hoa Mai': 'Hoa mai - Nở vào dịp Tết, màu vàng tươi sáng'
    # THÊM CÁC LOẠI HOA KHÁC TÙY THEO MODEL
}

# Layout
col_left, col_right = st.columns([0.45, 0.55])

with col_left:
    st.markdown("### > upload")
    uploaded_image = st.file_uploader("", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="", width=280)
        
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        image = image.resize(target_size)
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        if st.button("> predict"):
            input_name = session.get_inputs()[0].name
            predictions = session.run(None, {input_name: img_array})[0]
            
            idx = np.argmax(predictions[0])
            confidence = float(predictions[0][idx])
            flower_name = list(FLOWER_DATA.keys())[idx]
            
            st.markdown(f"### > {flower_name}")
            st.caption(f"confidence: {confidence:.2%}")
            
            st.markdown("> top 5")
            top5_idx = np.argsort(predictions[0])[-5:][::-1]
            for i, idx in enumerate(top5_idx, 1):
                prob = float(predictions[0][idx])
                name = list(FLOWER_DATA.keys())[idx]
                st.progress(prob, text=f"{i}. {name} - {prob:.2%}")

with col_right:
    st.markdown("### > supported classes")
    st.caption("cac loai hoa duoc ho tro | cuon de xem chi tiet")
    
    with st.container():
        for i, (flower, desc) in enumerate(FLOWER_DATA.items()):
            with st.expander(f"{i+1:02d}. {flower}"):
                st.caption(desc)

st.markdown("""
> version 1.0 2026 by Flower Recognition CNN
""")
