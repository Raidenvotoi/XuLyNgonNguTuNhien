import streamlit as st

# Khởi tạo bước trong session_state nếu chưa có
if "step" not in st.session_state:
    st.session_state["step"] = 1

st.title("Hướng dẫn từng bước")

if st.session_state["step"] == 1:
    st.subheader("Bước 1: Nhập thông tin")
    name = st.text_input("Nhập tên của bạn")
    if st.button("Hoàn thành bước 1"):
        st.session_state["step"] = 2
        st.rerun()

elif st.session_state["step"] == 2:
    st.subheader("Bước 2: Xác nhận thông tin")
    st.write(f"Bạn đã nhập: {st.session_state.get('name', 'Chưa có thông tin')}")
    if st.button("Hoàn thành bước 2"):
        st.session_state["step"] = 3
        st.rerun()

elif st.session_state["step"] == 3:
    st.subheader("Bước 3: Hoàn tất")
    st.success("Bạn đã hoàn thành tất cả các bước! 🎉")
    if st.button("Làm lại từ đầu"):
        st.session_state["step"] = 1
        st.rerun()
