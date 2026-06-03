import streamlit as st
import numpy as np
from PIL import Image
import io
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
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">flower recognition<span class="blinking-cursor">_</span></div>', unsafe_allow_html=True)

# Fake model state
if 'predict_count' not in st.session_state:
    st.session_state.predict_count = 0

FLOWER_INFO = {
    'Hoa Tulip': {
        'color': 'Do, hong, vang, tim, trang',
        'meaning': 'Tinh yeu hoan hao, su sang trong va quy phai',
        'origin': 'Trung A va Tho Nhi Ky'
    },
    'Hoa Bo Cong Anh': {
        'color': 'Vang',
        'meaning': 'Su lac quan, hy vong va suc manh tinh than',
        'origin': 'Vung on doi va cac vung nui cao'
    }
}

col_left, col_right = st.columns([0.5, 0.5])

with col_left:
    st.markdown("### upload")
    uploaded = st.file_uploader("", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    
    if uploaded is not None:
        img = Image.open(uploaded)
        st.image(img, width=250)
        
        if st.button("predict"):
            st.session_state.predict_count += 1
            
            # Luân phiên: lẻ -> Tulip, chẵn -> Bo Cong Anh
            if st.session_state.predict_count % 2 == 1:
                flower_name = 'Hoa Tulip'
                confidence = 0.94
                prob_tulip = 0.94
                prob_bca = 0.06
            else:
                flower_name = 'Hoa Bo Cong Anh'
                confidence = 0.96
                prob_tulip = 0.04
                prob_bca = 0.96
            
            flower = FLOWER_INFO[flower_name]
            
            st.markdown("---")
            st.markdown("xac suat tung loai hoa")
            st.progress(prob_tulip, text=f"Hoa Tulip: {prob_tulip:.2%}")
            st.progress(prob_bca, text=f"Hoa Bo Cong Anh: {prob_bca:.2%}")
            
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
st.caption("version 1.0 | flower recognition | demo mode")
