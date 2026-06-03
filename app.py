import streamlit as st
import numpy as np
from PIL import Image
import keras
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
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">> flower recognition<span class="blinking-cursor">_</span></div>', unsafe_allow_html=True)

MODEL_FILE = "flower.keras"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        return keras.models.load_model(MODEL_FILE)
    return None

model = load_model()

if model is None:
    st.error("> flower.keras not found")
    st.stop()

input_shape = model.input_shape
target_size = (input_shape[1], input_shape[2])

CLASS_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

FLOWER_INFO = {
    'daisy': {
        'name': 'Hoa Cuc Daisy',
        'origin': 'Chau Au va Bac Phi',
        'color': 'Trang, vang, hong',
        'meaning': 'Su ngay tho, trong sang va tinh yeu thuan khiet',
        'characteristic': 'Canh hoa mong manh, nhan hoa vang tuoi, thuong moc thanh cum'
    },
    'dandelion': {
        'name': 'Hoa Bo Cong Anh',
        'origin': 'Vung on doi va cac vung nui cao',
        'color': 'Vang',
        'meaning': 'Su lac quan, hy vong va suc manh tinh than',
        'characteristic': 'Hoa vang rucc, khi gia se chuyen thanh chum long trang bay theo gio'
    },
    'rose': {
        'name': 'Hoa Hong',
        'origin': 'Trung Quoc va Ba Tu',
        'color': 'Do, hong, trang, vang, cam',
        'meaning': 'Tinh yeu, dam me, su cao quy va long bien on',
        'characteristic': 'Canh hoa xep lop, huong thom dac trung, co gai tren canh'
    },
    'sunflower': {
        'name': 'Hoa Huong Duong',
        'origin': 'Bac My',
        'color': 'Vang',
        'meaning': 'Su trung thanh, may man va hanh phuc',
        'characteristic': 'Hoa lon huong ve phia mat troi, hat co the ep lay dau'
    },
    'tulip': {
        'name': 'Hoa Tulip',
        'origin': 'Trung A va Tho Nhi Ky',
        'color': 'Do, hong, vang, tim, trang',
        'meaning': 'Tinh yeu hoan hao, su sang trong va quy phai',
        'characteristic': 'Hoa hinh chuoc, canh day va mem, la hinh thon dai'
    }
}

st.markdown("""
> nhan dien 5 loai hoa pho bien
> buoc 1: chon file anh
> buoc 2: nhan nut predict
> buoc 3: xem ket qua
""")

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
        predictions = model.predict(img_array, verbose=0)[0]
        
        idx = np.argmax(predictions)
        confidence = float(predictions[idx])
        flower_key = CLASS_NAMES[idx]
        flower = FLOWER_INFO[flower_key]
        
        st.markdown("---")
        st.markdown(f"### > {flower['name']}")
        st.caption(f"confidence: {confidence:.2%}")
        
        st.markdown("---")
        st.markdown("> thong tin chi tiet")
        st.write(f"nguon goc: {flower['origin']}")
        st.write(f"mau sac: {flower['color']}")
        st.write(f"y nghia: {flower['meaning']}")
        st.write(f"dac diem: {flower['characteristic']}")
        
        st.markdown("---")
        st.markdown("> top 5")
        top5_idx = np.argsort(predictions)[-5:][::-1]
        for i, idx in enumerate(top5_idx, 1):
            prob = float(predictions[idx])
            name = FLOWER_INFO[CLASS_NAMES[idx]]['name']
            st.progress(prob, text=f"{i}. {name} - {prob:.2%}")

st.markdown("---")
st.caption("> version 1.0 | flower recognition cnn")
