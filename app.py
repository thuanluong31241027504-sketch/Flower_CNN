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
    .stApp {background-color: #ffffff;}
    * {font-family: 'Courier New', monospace;}
    @keyframes blink {0%,50%{opacity:1}51%,100%{opacity:0}}
    .blinking-cursor {animation: blink 1s step-end infinite; display: inline-block; width: 10px;}
    .main-title {
        color: #ff69b4;
        font-size: 2.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-align: center;
    }
    .stButton > button {
        background: #ff69b4 !important;
        color: white !important;
        border: none !important;
        border-radius: 0px !important;
        width: 100% !important;
        padding: 0.6rem !important;
    }
    .stButton > button:hover {
        background: #ff1493 !important;
    }
    .stProgress > div > div > div {
        background-color: #ff69b4;
    }
    hr {
        border-color: #ff69b4;
        opacity: 0.3;
    }
    .result-box {
        border: 2px solid #ff69b4;
        padding: 15px;
        margin-top: 15px;
        background-color: #fff0f5;
        text-align: center;
    }
    .result-box h2 {
        color: #ff69b4;
        font-size: 1.5rem;
        margin: 0;
    }
    .flower-card {
        border: 1px solid #ff69b4;
        padding: 10px;
        margin-bottom: 8px;
        background-color: #fff0f5;
    }
    .flower-title {
        color: #ff69b4;
        font-size: 0.9rem;
        font-weight: bold;
    }
    .flower-desc {
        color: #333;
        font-size: 0.65rem;
        line-height: 1.4;
    }
    .error-box {
        border: 2px solid #ff0000;
        background-color: #ffe0e0;
        padding: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">flower recognition<span class="blinking-cursor">_</span></div>', unsafe_allow_html=True)

# Thông tin các loài hoa
FLOWER_INFO = {
    'Hoa Cuc Daisy': {
        'color': 'Trang, vang, hong',
        'meaning': 'Su ngay tho, trong sang va tinh yeu thuan khiet',
        'origin': 'Chau Au va Bac Phi'
    },
    'Hoa Bo Cong Anh': {
        'color': 'Vang',
        'meaning': 'Su lac quan, hy vong va suc manh tinh than',
        'origin': 'Vung on doi va cac vung nui cao'
    },
    'Hoa Hong': {
        'color': 'Do, hong, trang, vang, cam',
        'meaning': 'Tinh yeu, dam me, su cao quy va long bien on',
        'origin': 'Trung Quoc va Ba Tu'
    },
    'Hoa Huong Duong': {
        'color': 'Vang',
        'meaning': 'Su trung thanh, may man va hanh phuc',
        'origin': 'Bac My'
    },
    'Hoa Tulip': {
        'color': 'Do, hong, vang, tim, trang',
        'meaning': 'Tinh yeu hoan hao, su sang trong va quy phai',
        'origin': 'Trung A va Tho Nhi Ky'
    }
}

# Fake model - tạo output giả để demo
def fake_predict():
    if 'counter' not in st.session_state:
        st.session_state.counter = 0
    
    st.session_state.counter += 1
    # Luân phiên giữa các loài hoa
    if st.session_state.counter % 5 == 1:
        return 4, 0.92  # Tulip
    elif st.session_state.counter % 5 == 2:
        return 1, 0.88  # Bo Cong Anh
    elif st.session_state.counter % 5 == 3:
        return 2, 0.85  # Hong
    elif st.session_state.counter % 5 == 4:
        return 3, 0.90  # Huong Duong
    else:
        return 0, 0.87  # Daisy

col_left, col_right = st.columns([0.5, 0.5])

with col_left:
    st.markdown("### upload")
    uploaded = st.file_uploader("", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    
    if uploaded is not None:
        img = Image.open(uploaded)
        st.image(img, width=250)
        
        if st.button("predict"):
            # Giả lập dự đoán
            idx, confidence = fake_predict()
            display_names = list(FLOWER_INFO.keys())
            flower_name = display_names[idx]
            flower = FLOWER_INFO[flower_name]
            
            # Tạo xác suất giả cho các loài
            probs = [0.03, 0.03, 0.03, 0.03, 0.03]
            probs[idx] = confidence
            # Điều chỉnh các xác suất khác
            remaining = 1.0 - confidence
            for i in range(5):
                if i != idx:
                    probs[i] = remaining / 4
            
            st.markdown("---")
            st.markdown("xac suat tung loai hoa")
            for i, name in enumerate(display_names):
                st.progress(probs[i], text=f"{name}: {probs[i]:.2%}")
            
            st.markdown(f"""
            <div class="result-box">
                <h2>{flower_name}</h2>
                <p>do tin cay: {confidence:.2%}</p>
                <hr>
                <p><b>mau sac:</b> {flower['color']}</p>
                <p><b>y nghia:</b> {flower['meaning']}</p>
                <p><b>nguon goc:</b> {flower['origin']}</p>
            </div>
            """, unsafe_allow_html=True)

with col_right:
    st.markdown("### thu vien hoa")
    
    for name, flower in FLOWER_INFO.items():
        with st.expander(name):
            st.markdown(f"""
            <div class="flower-card">
                <div class="flower-title">{name}</div>
                <div class="flower-desc"><b>mau sac:</b> {flower['color']}</div>
                <div class="flower-desc"><b>y nghia:</b> {flower['meaning']}</div>
                <div class="flower-desc"><b>nguon goc:</b> {flower['origin']}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.caption("version 1.0 | flower recognition cnn")
