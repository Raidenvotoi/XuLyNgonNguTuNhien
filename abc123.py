import streamlit as st
import time

st.title("Tiến trình từng bước")

progress = st.progress(0)

steps = ["Chuẩn bị dữ liệu", "Tiền xử lý dữ liệu", "Hiển thị kết quả",]

for i, step in enumerate(steps):
    st.write(f"### Bước {i+1}: {step}")
    time.sleep(1)  # Giả lập quá trình xử lý
    progress.progress((i + 1) / len(steps))
