import streamlit as st
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Load NLP model của spaCy (tiếng Anh)
nlp = spacy.load("en_core_web_sm")

# Kiểm tra nếu `st.session_state.df3` chưa có dữ liệu
if "df3" not in st.session_state:
    st.session_state.df3 = pd.DataFrame({"text": [
        "Machine learning helps computers learn patterns.",
        "Natural Language Processing (NLP) is a subset of AI.",
        "Python is widely used for data science and AI."
    ]})

st.title("Trực Quan Cây Cú Pháp (Dependency Parsing)")

# Hiển thị dữ liệu
st.subheader("Dữ liệu văn bản:")
st.write(st.session_state.df3)

# Chọn câu để vẽ
selected_text = st.selectbox("Chọn câu để phân tích:", st.session_state.df3["text"])

# Phân tích cú pháp với spaCy
doc = nlp(selected_text)

# 1️⃣ Tạo đồ thị
G = nx.DiGraph()

# 2️⃣ Thêm các từ vào đồ thị
for token in doc:
    G.add_edge(token.head.text, token.text, label=token.dep_)

# 3️⃣ Vẽ đồ thị
st.subheader("Graph Network - Cây Cú Pháp")

fig, ax = plt.subplots(figsize=(8, 5))
pos = nx.spring_layout(G)  # Bố cục của đồ thị
nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=2000, font_size=10, font_weight="bold", ax=ax)

# Hiển thị nhãn dependency trên các cạnh
edge_labels = {(token.head.text, token.text): token.dep_ for token in doc}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red", ax=ax)

# Hiển thị đồ thị trên Streamlit
st.pyplot(fig)

# Hiển thị chi tiết dependency tree
st.subheader("Chi tiết Dependency Parsing:")
for token in doc:
    st.write(f"**{token.text}** ← ({token.dep_}) ← **{token.head.text}**")
