import streamlit as st
import re
from io import StringIO
# Import necessary modules
import pandas as pd
import requests
from bs4 import BeautifulSoup
from re import findall
import time
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import contractions
from spellchecker import SpellChecker
import nltk
from nltk import pos_tag, word_tokenize, RegexpParser
from nltk import pos_tag
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from IPython.display import display
import string
from googletrans import Translator
import random
from langdetect import detect, detect_langs
from collections import Counter
import seaborn as sns
import spacy
import networkx as nx
# from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
# import svgling # Import svgling for visualization


def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text.lower())  # Chuyển về chữ thường
    filtered_text = " ".join(
        word for word in word_tokens if word not in stop_words)
    return filtered_text


# def clean_text(input_text):
#     # 1. Xóa dấu chấm câu Xóa biểu tượng (ví dụ: @, #, $, %, ...)
#     input_text = re.sub(r'[^\w\s]', '', input_text)

#     # 2. Xóa số
#     input_text = re.sub(r'\d+', '', input_text)

#     # 3. Xóa khoảng trắng dư thừa
#     input_text = re.sub(r'\s+', ' ', input_text).strip()
#     return re.sub(r'[^a-zA-Z0-9\s]', '', input_text).strip()


def get_comment(soup):
    soupcmt = soup
    article_cmt = soupcmt.find_all('article', 'user-review-item')
    cmt = []
    # print(article_cmt)
    for i in article_cmt:
        try:
            # print("hi")
            title = i.find("h3", "ipc-title__text").text
            # print(title)
            content = i.find("div", "ipc-html-content-inner-div").text
            star = i.find("span", "ipc-rating-star--rating").text
            cmt.append([title, star, content])
        except:
            continue
    return cmt


def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result


def remove_whitespace(text):
    return " ".join(text.split())


def remove_punctuation(text):
    # translator = str.maketrans('', '', string.punctuation)
    return text.translate(str.maketrans("", "", string.punctuation))


def clean_text(text):
    if clean_lower:
        text = text.lower()
    if clean_ExpandingContractions:
        text = contractions.fix(text)
    if clean_PunctuationRemoval:
        text = text.translate(str.maketrans("", "", string.punctuation))

    if clean_numbers:
        text = re.sub(r"\d+", "", text)

    if clean_spaces:
        text = " ".join(text.split())

    if clean_StopWordsRemoval:
        # print("xoa tu dung ")
        stop_words = set(stopwords.words("english"))
        words = word_tokenize(text)
        text = " ".join(word for word in words if word not in stop_words)
        # print(text)
    if clean_symbols:
        text = re.sub(r"[^\w\s]", "", text)

    if clean_Stemming:
        stemmer = PorterStemmer()
        words = word_tokenize(text)
        text = " ".join(stemmer.stem(word) for word in words)

    if clean_Lemmatization:
        lemmatizer = WordNetLemmatizer()
        words = word_tokenize(text)
        text = " ".join(lemmatizer.lemmatize(word) for word in words)

    if clean_specialchar:
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    return text


st.title("Xử lý Văn Bản")


def crawl_text(limit):
    user_agent = "checkvote"
    token="eyJhbGciOiJSUzI1NiIsImtpZCI6IlNIQTI1NjpzS3dsMnlsV0VtMjVmcXhwTU40cWY4MXE2OWFFdWFyMnpLMUdhVGxjdWNZIiwidHlwIjoiSldUIn0.eyJzdWIiOiJ1c2VyIiwiZXhwIjoxNzQxMjI5ODE2LjcyMTE2LCJpYXQiOjE3NDExNDM0MTYuNzIxMTYsImp0aSI6Ii1UTjBrQ2o3Qzd0YjdmZEdqNlFyb25BcmxMckg4USIsImNpZCI6ImRocnM3NkZIbUdDWWFGZVNWZ0FUcFEiLCJsaWQiOiJ0Ml80MGRxd29hNiIsImFpZCI6InQyXzQwZHF3b2E2IiwibGNhIjoxNTYxNDUxOTE2ODYwLCJzY3AiOiJlSnlLVnRKU2lnVUVBQURfX3dOekFTYyIsImZsbyI6OX0.HgPghVzn0jeg1I5U4gaTl5hnYceoZ-lrKDCBQpq-GPaXlEls6Z41ziYbU0hZbC-BSr3cGjw5-TJ0WdyaaFkVVJxiNCv_TkqFyHrCQDcTWmgGAqg3indBjX8YrojjiXDzAeEK1_UlDsYfHwSzxwjOkZIrR7IUjfAWXHyzMqxOG4eRVe0QCm1jvEXlgIeFaRRqNxWxk-Bvy-OT9bHlc-kCbIRTEh6QpIj-W6uAAbNFIxjryUahuFO4HWn7X_yamGednbq2OjOLDZ5_MeQx044N1Rc2HGnjiDqkB-A4aihqvygRh7BLk-FKm2RfhupaHW-6JLafbKDSLivzungWk2PgPA"
    headers = {
        "Authorization": f"bearer {token}",
        "User-Agent": user_agent
    }
    params = {"limit": limit, "t": "week"}
    response = requests.get(
        "https://oauth.reddit.com/r/shortstories/top",
        headers=headers,
        params=params,
    )
    ups = []
    text = []
    title = []
    posts = response.json()

    for i in range(len(posts["data"]["children"])):

        text.append(posts["data"]["children"][i]['data']
                    ['selftext'].replace('\r', '').replace('\n', ' '))
        ups.append(posts["data"]["children"][i]['data']['ups'])
        title.append(posts["data"]["children"][i]['data']['title'])
    data = {'title': title, 'ups': ups, 'text': text, }

    return pd.DataFrame.from_dict(data)


if "text" not in st.session_state:
    st.session_state.text = []
option = st.radio("Chọn phương thức nhập văn bản:", [
                  "Nhập số để crawl văn bản", "Tải lên file .txt"])
if option == "Tải lên file .txt":
    uploaded_file = st.file_uploader("Chọn tệp để tải lên", type=["txt"])
    if uploaded_file is not None:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        text = stringio.read()
        st.session_state.text = text
        # st.text_area("Nội dung tệp:", text, height=150)
elif option == "Nhập số để crawl văn bản":
    number = st.number_input("Nhập một số:", min_value=1,
                             step=1, max_value=50, value=3)
    if st.button("Crawl"):
        df = crawl_text(number)
        st.dataframe(df)
        st.session_state.df = df
        # st.text_area("Văn bản đã crawl:", str(text), height=150)

# txt_areaarea = st.text_area("Nhập văn bản...")
st.title("B2 Tăng cường dữ liệu")


def synonym_replacement_EN(text, n=2):
    # text=df['text']
    words = word_tokenize(text)
    new_words = words.copy()
    random_words = list(set(words))
    random.shuffle(random_words)

    num_replaced = 0
    for word in random_words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            if synonym != word:
                new_words = [synonym if w == word else w for w in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break
    return ' '.join(new_words)


def word_shuffling(text):
    words = text.split()
    random.shuffle(words)
    return ' '.join(words)


def noise_injection(text, probability=0.1):
    words = list(text)  # Chuyển chuỗi thành danh sách ký tự
    for i in range(len(words)):  # Duyệt từng ký tự
        if random.random() < probability:  # Xác suất thay thế ký tự
            # Thay bằng ký tự ngẫu nhiên
            words[i] = random.choice('abcdefghijklmnopqrstuvwxyz')
    return ''.join(words)  # Ghép lại thành chuỗi


def random_word_deletion(text, probability=0.2):
    words = text.split()
    new_words = [word for word in words if random.random() > probability]
    if len(new_words) == 0:  # Ensure at least one word remains
        return random.choice(words)
    return ' '.join(new_words)


async def back_translation(text, lang='fr'):

    translator = Translator()
    translated = await translator.translate(text, dest=lang)
    back_translated = await translator.translate(translated.text, src=lang, dest=detect(text))
    return back_translated.text
# def tangcuong(df, nn)
b2op = st.radio("Chọn phương pháp tăng cường", ["Thay thế từ đồng nghĩa",
                                                "Xáo trộn từ",
                                                "Thêm nhiễu",
                                                "Xóa từ ngẫu nhiên"], horizontal=True)
num_row = st.number_input("Nhập số câu cần tăng cường", min_value=4)


if b2op == "Thay thế từ đồng nghĩa":
    st.write("Bạn đã chọn phương pháp thay thế từ đồng nghĩa.")
    # synonym_replacement_EN(st.session_state.df)
    num_dongnghia = st.number_input("so tu thay the tren moi bai", min_value=4)


elif b2op == "Xáo trộn từ":
    st.write("Bạn đã chọn phương pháp xáo trộn từ.")
    # Thêm logic xử lý tại đây

elif b2op == "Thêm nhiễu":
    st.write("Bạn đã chọn phương pháp thêm nhiễu.")
    num_xacsuat = st.number_input(
        label="Nhập xác suất nhiễu", min_value=0.1, max_value=1.00, step=0.05)

elif b2op == "Xóa từ ngẫu nhiên":
    st.write("Bạn đã chọn phương pháp xóa từ ngẫu nhiên.")
    # Thêm logic xử lý tại đây
    num_xacsuat = st.number_input(
        label="Nhập xác suất nhiễu", min_value=0.1, max_value=1.00, step=0.05)





b2btn = st.button("xử lý tang cuong")
if b2btn:
    if "df" not in st.session_state:
        st.error("Vui lòng crawl dữ liệu trước khi tăng cường!")
    else:
        new_rows = []
        for i in range(num_row):
            random_num = random.randint(0, len(st.session_state.df) - 1)
            row = st.session_state.df.loc[random_num].copy()
            if b2op == "Thay thế từ đồng nghĩa":
                row['text'] = synonym_replacement_EN(row['text'], num_dongnghia)
            elif b2op == "Xáo trộn từ":
                row['text'] = word_shuffling(row['text'])
            elif b2op == "Thêm nhiễu":
                row['text'] = noise_injection(row['text'], num_xacsuat)
            elif b2op == "Xóa từ ngẫu nhiên":
                row['text'] = random_word_deletion(row['text'], num_xacsuat)
            new_rows.append(row)
        st.session_state.df = pd.concat(
            [st.session_state.df, pd.DataFrame(new_rows)], ignore_index=True
        )
        st.dataframe(st.session_state.df)
if "df" in st.session_state :
    st.dataframe(st.session_state.df)
    st.write("So luong entry: ",len(st.session_state.df))
# txt_areaarea = st.text_area("Văn bản...",st.session_state.text)
st.title("Bước 3 Xử lý tiền dữ liệu")
col1, col2, col3 = st.columns(3)

with col1:
    clean_lower = st.checkbox("Chuyển đổi chữ thường")
    clean_numbers = st.checkbox("Xóa số")
    clean_symbols = st.checkbox("Xóa các biểu tượng")
    clean_ExpandingContractions = st.checkbox("Xử lý từ viết tắt")

with col2:
    clean_PunctuationRemoval = st.checkbox("Loại bỏ dấu câu")
    clean_spaces = st.checkbox("Loại bỏ khoảng trắng thừa")
    clean_specialchar = st.checkbox("Nhận diện và loại bỏ ký tự đặc biệt")

with col3:
    clean_StopWordsRemoval = st.checkbox("Xóa từ dừng")
    clean_Stemming = st.checkbox("Cắt gốc từ Stemming")
    clean_Lemmatization = st.checkbox("Chuẩn hóa từ về từ điển Lemmatization")

if st.button("Xử lý"):
    st.session_state.df2 = st.session_state.df.copy()
    st.session_state.df2["text"] = st.session_state.df2["text"].apply(
        clean_text)
    st.dataframe(st.session_state.df2)
if "df2" in st.session_state :
    st.dataframe(st.session_state.df2)
    st.write("So luong entry: ",len(st.session_state.df2))
st.title("Bước 4 Text Representation")

option_pre = st.radio("Chọn cách xử lý văn bản:",
                      ["NER để hiển thị thực thể",
                       "Tokenization",
                       "POS tagging",
                       "Parsing",
                       "One-Hot Encoding",
                       "Bag of Words (BoW)",
                       "Bag of N-Grams",
                       "TF-IDF"
                       ], horizontal=True)

from wordcloud import WordCloud
import matplotlib.pyplot as plt
nlp = spacy.load("en_core_web_sm")
if option_pre=="Parsing" or \
        option_pre == "NER để hiển thị thực thể"\
        and "df2" in st.session_state :
    selected_text = st.selectbox("Chọn bài để phân tích:", st.session_state.df2["text"])
    
    # Phân tích cú pháp với spaCy
    if  str(selected_text) !="" :
        doc = nlp(str(selected_text))
        sentences = list(doc.sents)
        selected_sent=st.selectbox("Chọn câu để phân tích",[str(sent) for sent in sentences])
if option_pre =="One-Hot Encoding"  or option_pre =="Bag of Words (BoW)" or option_pre =="TF-IDF":
    selected_text = st.selectbox("Chọn bài để phân tích:", st.session_state.df2["text"])
    if  str(selected_text) !="" :
        doc = nlp(str(selected_text))
        sentences = list(doc.sents)
elif option_pre=="Bag of N-Grams":
    selected_text = st.selectbox("Chọn bài để phân tích:", st.session_state.df2["text"])
    if  str(selected_text) !="" :
        doc = nlp(str(selected_text))
        sentences = list(doc.sents)
    start_gram=st.number_input("start_gram",value=2)
    end_gram=st.number_input("end_gram",value=2)

if st.button("Xử lý biểu diễn"):
    st.session_state.df3 = st.session_state.df2.copy()

    if option_pre == "NER để hiển thị thực thể":
        doc_sent = nlp(str(selected_sent))

        # 1️⃣ Tạo danh sách thực thể
        entities = [{"Thực thể": ent.text, "Loại": ent.label_} for ent in doc_sent.ents]

        # 2️⃣ Chuyển thành DataFrame
        df_entities = pd.DataFrame(entities)
        if not df_entities.empty:
            st.table(df_entities)
        else:
            st.write("Không tìm thấy thực thể nào!")

    elif option_pre == "Tokenization":
        st.title("Tạo Word Cloud từ DataFrame")

        # Hiển thị DataFrame
        st.subheader("Dữ liệu văn bản:")
        st.write(st.session_state.df3)

        # 1️⃣ Gộp toàn bộ văn bản trong cột "text" thành một chuỗi
        text_data = " ".join(st.session_state.df3["text"]).lower()  # Chuyển về chữ thường

        # 2️⃣ Tokenization
        tokens = word_tokenize(text_data)

        # 3️⃣ Xóa dấu câu
        tokens = [word for word in tokens if word not in string.punctuation]

        # 4️⃣ Gộp danh sách tokens thành chuỗi
        processed_text = " ".join(tokens)

        # 5️⃣ In kết quả kiểm tra
        # st.write("📌 Văn bản sau khi xử lý:")
        # st.write(processed_text)

        # 6️⃣ Tạo & hiển thị Word Cloud
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(processed_text)

        st.subheader("Word Cloud")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    elif option_pre == "POS tagging":
        text_data = " ".join(st.session_state.df3["text"]).lower()

        # 2️⃣ Tokenization
        tokens = word_tokenize(text_data)

        # 3️⃣ POS Tagging
        pos_tags = nltk.pos_tag(tokens)

        # 4️⃣ Đếm số lượng từng loại từ
        pos_counts = Counter(tag for _, tag in pos_tags)

        # 5️⃣ Chuyển dữ liệu thành DataFrame để vẽ biểu đồ
        df_pos = pd.DataFrame(pos_counts.items(), columns=["POS Tag", "Count"])
        df_pos = df_pos.sort_values(by="Count", ascending=False)  # Sắp xếp giảm dần

        # 6️⃣ Vẽ Bar Chart với Seaborn
        st.subheader("Bar Chart - Phân loại từ loại (POS)")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=df_pos["POS Tag"], y=df_pos["Count"], palette="viridis", ax=ax)
        ax.set_xlabel("POS Tag")
        ax.set_ylabel("Số lượng")
        ax.set_title("Tần suất các từ loại trong văn bản")
        st.pyplot(fig)

        # # 7️⃣ Hiển thị dữ liệu POS để kiểm tra
        # st.subheader("Chi tiết POS Tagging")
        # st.write(pos_tags)

    elif option_pre == "Parsing":
   
        
        
        doc_sent = nlp(str(selected_sent))
        # 1️⃣ Tạo đồ thị
        G = nx.DiGraph()

        # 2️⃣ Thêm các từ vào đồ thị
        for token in doc_sent:
            G.add_edge(token.head.text, token.text, label=token.dep_)

        # 3️⃣ Vẽ đồ thị
        st.subheader("Graph Network - Cây Cú Pháp")

        fig, ax = plt.subplots(figsize=(10, 10))
        pos = nx.spring_layout(G)  # Bố cục của đồ thị
        nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=2000, font_size=10, font_weight="bold", ax=ax)

        # Hiển thị nhãn dependency trên các cạnh
        edge_labels = {(token.head.text, token.text): token.dep_ for token in doc_sent}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red", ax=ax)

        # Hiển thị đồ thị trên Streamlit
        st.pyplot(fig)

        # Hiển thị chi tiết dependency tree
        st.subheader("Chi tiết Dependency Parsing:")
        dic={'Từ hiện tại trong câu':[token.text for token in doc_sent],
             'Quan hệ cú pháp':[token.dep_ for token in doc_sent],
             'Từ gốc':[token.head.text for token in doc_sent],
             }
        st.dataframe(pd.DataFrame.from_dict(dic))
        # for token in doc_sent:
        #     st.write(f"**{token.text}** ← ({token.dep_}) ← **{token.head.text}**")

    elif option_pre=="One-Hot Encoding":
        count_vect = CountVectorizer(binary=True)
        bow_rep = count_vect.fit_transform([str(span) for span in sentences])
        # print (bow_rep.toarray())
        # print(count_vect.get_feature_names_out())
        df = pd.DataFrame(bow_rep.toarray(), columns=count_vect.get_feature_names_out(),index=[str(span) for span in sentences])
        st.dataframe(df)
    
    elif option_pre=="Bag of Words (BoW)":
        count_vect = CountVectorizer()
        bow_rep = count_vect.fit_transform([str(span) for span in sentences])

        df3 = pd.DataFrame(bow_rep.toarray(), columns=count_vect.get_feature_names_out(),index=[str(span) for span in sentences])
        st.dataframe(df3)
    elif option_pre=="Bag of N-Grams":
        count_vect = CountVectorizer(ngram_range=(start_gram, end_gram))
        bow_rep = count_vect.fit_transform([str(span) for span in sentences])
        df3 = pd.DataFrame(bow_rep.toarray(), columns=count_vect.get_feature_names_out(), index=[str(span) for span in sentences])
        st.dataframe(df3)
    elif option_pre=="TF-IDF":
        count_vect = TfidfVectorizer()
        bow_rep = count_vect.fit_transform([str(span) for span in sentences])

        df3 = pd.DataFrame(bow_rep.toarray(), columns=count_vect.get_feature_names_out(),index=[str(span) for span in sentences])
        st.dataframe(df3)
    # result = st.text_area(label="ket qua", value=t)


