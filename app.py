import streamlit as st
import os

st.title("Test Model ONNX")

# Kiểm tra file có tồn tại không
model_path = "flowerpro.onnx"

if os.path.exists(model_path):
    st.success(f"✅ Tìm thấy file: {model_path}")
    file_size = os.path.getsize(model_path)
    st.write(f"Kích thước: {file_size / 1024 / 1024:.2f} MB")
else:
    st.error(f"❌ Không tìm thấy file: {model_path}")
    st.write("Danh sách file trong thư mục:")
    for f in os.listdir('.'):
        st.write(f"  - {f}")
