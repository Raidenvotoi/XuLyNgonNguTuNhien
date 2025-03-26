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
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import praw
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertModel
from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier  # Nếu muốn dùng cây quyết định
from sklearn.ensemble import RandomForestClassifier  # Đúng cách
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
import tensorflow_hub as hub
import tensorflow as tf

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text.lower())  # Chuyển về chữ thường
    filtered_text = " ".join(
        word for word in word_tokens if word not in stop_words)
    return filtered_text
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
def crawl_text(limit,subreddit):
    client_id = "cfgRNdcf6Vw7PF9mBKrWKA"
    client_secret = "k8YD6GloeBp5uSvV0VHOMknpyjjIrA"
    user_agent = "conheo"
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )
    subreddit = reddit.subreddit(subreddit)
    u=0
    ups = []
    text = []
    title = []
    for submission in subreddit.hot(limit=limit):  
        text.append(submission.selftext.replace('\r', '').replace('\n', ' '))
        ups.append(submission.ups)
        title.append(submission.title)

    data = {'title': title, 'ups': ups, 'text': text, }

    return pd.DataFrame.from_dict(data)

def predict_sentiment(text,pipline_cls):
    
    label = pipline_cls.predict([text])[0]
    return "Tích cực" if label == 1 else "Tiêu cực"
def predict_topic(text,pipline_cls):
    
    label = pipline_cls.predict([text])[0]
    if label == 0:  
        label = "World"
    elif label == 1:
        label = "Sports"
    elif label == 2:
        label = "Business"
    elif label == 3:
        label = "Sci/Tech"
    return label
def predict_question(text,pipline_cls):
    
    label = pipline_cls.predict([text])[0]
    if label==0:
        label="Viết tắt"
    elif label==1:
        label="Thực thể"
    elif label==2:
        label="Mô tả và khái niệm trừu tượng."
    elif label==3:
        label="Con người"
    elif label==4:
        label="Vị trí"
    elif label==5:
        label="Giá trị số"
        
    return label
# if "df3" in st.session_state:
st.title("Xử lý ngôn ngữ tự nhiên")



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

def tokenize(corpus):
    words = [word.lower() for sentence in corpus for word in sentence.split()]
    vocab = list(set(words))
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    ix_to_word = {i: word for word, i in word_to_ix.items()}
    return words, vocab, word_to_ix, ix_to_word

def create_cbows(words, word_to_ix, window_size=2):
    data = []
    for i in range(window_size, len(words) - window_size):
        context = [words[i - j] for j in range(1, window_size + 1)] + \
                [words[i + j] for j in range(1, window_size + 1)]
        target = words[i]
        data.append((context, target))
    return data
def create_skipgrams(words, word_to_ix, window_size=2):
    data = []
    for i in range(window_size, len(words) - window_size):
        target = words[i]
        context = [words[i - j] for j in range(1, window_size + 1)] + \
                  [words[i + j] for j in range(1, window_size + 1)]
        for ctx in context:
            data.append((word_to_ix[target], word_to_ix[ctx]))  # (target, context)
    return data

def build_vocab(corpus):
    words = [word.lower() for sentence in corpus for word in sentence.split()]
    vocab = list(set(words))
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    ix_to_word = {i: word for word, i in word_to_ix.items()}
    return vocab, word_to_ix, ix_to_word
def build_cooccurrence_matrix(corpus, word_to_ix, window_size=2):
    matrix = np.zeros((vocab_size, vocab_size))
    
    for sentence in corpus:
        words = sentence.lower().split()
        for i, word in enumerate(words):
            word_idx = word_to_ix[word]
            for j in range(1, window_size + 1):
                if i - j >= 0:
                    context_idx = word_to_ix[words[i - j]]
                    matrix[word_idx, context_idx] += 1
                if i + j < len(words):
                    context_idx = word_to_ix[words[i + j]]
                    matrix[word_idx, context_idx] += 1

    return matrix
class GloVe:
    def __init__(self, vocab_size, embedding_dim=10, alpha=0.75, x_max=100):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.x_max = x_max

        # Khởi tạo vector từ và bias
        self.W = np.random.rand(vocab_size, embedding_dim)
        self.W_tilde = np.random.rand(vocab_size, embedding_dim)
        self.b = np.zeros(vocab_size)
        self.b_tilde = np.zeros(vocab_size)

    def weight_function(self, x):
        return (x / self.x_max) ** self.alpha if x < self.x_max else 1

    def train(self, cooccurrence_matrix, epochs=100, learning_rate=0.05):
        nonzero_indices = np.nonzero(cooccurrence_matrix)

        for epoch in range(epochs):
            total_loss = 0
            for i, j in zip(*nonzero_indices):
                X_ij = cooccurrence_matrix[i, j]
                weight = self.weight_function(X_ij)

                # Tính toán lỗi loss
                loss_ij = (self.W[i].dot(self.W_tilde[j]) + self.b[i] + self.b_tilde[j] - np.log(X_ij)) ** 2
                total_loss += weight * loss_ij

                # Cập nhật vector từ và bias
                gradient = 2 * weight * (self.W[i].dot(self.W_tilde[j]) + self.b[i] + self.b_tilde[j] - np.log(X_ij))

                self.W[i] -= learning_rate * gradient * self.W_tilde[j]
                self.W_tilde[j] -= learning_rate * gradient * self.W[i]
                self.b[i] -= learning_rate * gradient
                self.b_tilde[j] -= learning_rate * gradient

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")




if "step" not in st.session_state:
    st.session_state.step = 1
    
    
progress = st.progress(0)   
progress.progress(st.session_state.step/5)
    
if st.session_state.step == 1:
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('wordnet')
    st.header("Bước 1 Thu thập dữ liệu ")
    if "text" not in st.session_state:
        st.session_state.text = []
        
    option_sub=st.selectbox("Chọn subredit để crawl", [
                                            "shortscarystories",
                                            "nosleep",
                                            "creepypasta",
                                            "news",]
                            )
    if option_sub=="shortscarystories":
        st.write("Truyện ngắn kinh dị - Ngôi nhà của truyện kinh dị Flash Fiction. Chúng tôi mang đến những câu chuyện kinh dị, hồi hộp và những tình tiết đau lòng trong 500 từ hoặc ít hơn")
    elif option_sub=="creepypasta":
        st.write("r/Creepypasta | Nơi dành cho những người hâm mộ truyện Creepypasta.")
    elif option_sub=="nosleep":
        st.write("Nosleep là nơi để người dùng Reddit chia sẻ những trải nghiệm cá nhân đáng sợ của họ.")
    elif option_sub=="news":
        st.write("Nơi đăng tải các bài viết về các sự kiện hiện tại ở Hoa Kỳ và phần còn lại của thế giới. Thảo luận tất cả ở đây.")
    option = st.radio("Chọn phương thức nhập văn bản:", [
                    "Nhập số để lấy văn bản",
                    "Tải lên file .txt"
                    ])
    if option == "Tải lên file .txt":
        uploaded_file = st.file_uploader("Chọn tệp để tải lên", type=["txt"])
        if uploaded_file is not None:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            text = stringio.read()
            # st.session_state.text = textVanw
            data = {'title': ["Văn bản của người dùng"], 'ups': [0], 'text': text, }
            st.session_state.df = pd.DataFrame.from_dict(data)
            # st.text_area("Nội dung tệp:", text, height=150)
    elif option == "Nhập số để lấy văn bản":
        number = st.number_input("Nhập một số:", min_value=1,
                                step=1, max_value=50, value=3)
        if st.button("Crawl"):
            df = crawl_text(number,option_sub)
            st.dataframe(df)
            st.session_state.df = df
            # st.text_area("Văn bản đã crawl:", str(text), height=150)
        

elif st.session_state.step == 2:
    st.header("Bước 2 Tăng cường dữ liệu")

    b2op = st.radio("Chọn phương pháp tăng cường", ["Thay thế từ đồng nghĩa",
                                                    "Xáo trộn từ",
                                                    "Thêm nhiễu",
                                                    "Xóa từ ngẫu nhiên"], horizontal=True)
    num_row = st.number_input("Nhập số bài viết cần tăng cường", min_value=4)


    if b2op == "Thay thế từ đồng nghĩa":
        st.write("Bạn đã chọn phương pháp thay thế từ đồng nghĩa.")
        # synonym_replacement_EN(st.session_state.df)
        num_dongnghia = st.number_input("Nhập số từ thay thế trên mỗi bài", min_value=4)


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





    b2btn = st.button("Tăng cường")
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
        pass
        # st.dataframe(st.session_state.df)
        # st.write("So luong entry: ",len(st.session_state.df))

elif st.session_state.step == 3:
    st.header("Bước 3 Xử lý tiền dữ liệu")
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('wordnet')
    nlp = spacy.load("en_core_web_sm")
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
        # st.dataframe(st.session_state.df2)
        st.write("So luong entry: ",len(st.session_state.df2))
        


    option_3_2 =st.radio("Chọn cách xử lý văn bản:",["NER để hiển thị thực thể",
                        "Tokenization",
                        "POS tagging",
                        "Parsing"],horizontal=True)
    if option_3_2=="Parsing" or \
            option_3_2 == "NER để hiển thị thực thể"\
            and "df2" in st.session_state :
        selected_text = st.selectbox("Chọn bài xử lý:", st.session_state.df2["text"])

        # Phân tích cú pháp với spaCy
        if  str(selected_text) !="" :
            doc = nlp(str(selected_text))
            sentences = list(doc.sents)
            selected_sent=st.selectbox("Chọn câu xử lý",[str(sent) for sent in sentences])
    if st.button("xu ly 3.2"):
        st.session_state.df3 = st.session_state.df2.copy()
        if option_3_2 == "NER để hiển thị thực thể":
            doc_sent = nlp(str(selected_sent))

            # 1️⃣ Tạo danh sách thực thể
            entities = [{"Thực thể": ent.text, "Loại": ent.label_} for ent in doc_sent.ents]

            # 2️⃣ Chuyển thành DataFrame
            df_entities = pd.DataFrame(entities)
            if not df_entities.empty:
                st.table(df_entities)
            else:
                st.write("Không tìm thấy thực thể nào!")

        elif option_3_2 == "Tokenization":
            # st.title("Tạo Word Cloud từ DataFrame")

            # Hiển thị DataFrame
            st.subheader("Dữ liệu văn bản:")
            st.write(st.session_state.df2)

            # 1️⃣ Gộp toàn bộ văn bản trong cột "text" thành một chuỗi
            text_data = " ".join(st.session_state.df2["text"]).lower()  # Chuyển về chữ thường

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

        elif option_3_2 == "POS tagging":
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
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

            st.pyplot(fig)

            # # 7️⃣ Hiển thị dữ liệu POS để kiểm tra
            # st.subheader("Chi tiết POS Tagging")
            # st.write(pos_tags)

        elif option_3_2 == "Parsing":
    
            
            
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

elif st.session_state.step == 4:
    
    nlp = spacy.load("en_core_web_sm")
    elmo = hub.load("https://tfhub.dev/google/elmo/3")
    st.header("Bước 4 Text Representation")
    # st.title("Bước 5 Text Distributed Representations")
    option_pre = st.radio("Chọn cách xử lý văn bản:",
                        [
                        "One-Hot Encoding",
                        "Bag of Words (BoW)",
                        "Bag of N-Grams",
                        "TF-IDF",
                        "CBOW",
                        "skip gram",
                        "GloVe",
                        "ELMo",
                            "BERT"
                        ], horizontal=True)

    if "df2" in st.session_state:

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
        elif option_pre=="CBOW":
            selected_text = st.selectbox("Chọn bài để phân tích CBOW:", st.session_state.df2["text"])
            if  str(selected_text) !="" :
                doc = nlp(str(selected_text))
                sentences = list(doc.sents)
                # selected_sent=st.selectbox("Chọn câu xử lý CBOW",[str(sent) for sent in sentences])
                corpus=[str(sent) for sent in sentences]
                words, vocab, word_to_ix, ix_to_word = tokenize(corpus)
                a=list(word_to_ix.keys())
                a.sort()
                testword=st.selectbox("Chọn từ để kiểm tra",a)
        elif option_pre=="skip gram":
            selected_text = st.selectbox("Chọn bài để phân tích skip gram:", st.session_state.df2["text"])
            if  str(selected_text) !="" :
                doc = nlp(str(selected_text))
                sentences = list(doc.sents)
                # selected_sent=st.selectbox("Chọn câu xử lý CBOW",[str(sent) for sent in sentences])
                corpus=[str(sent) for sent in sentences]
                words, vocab, word_to_ix, ix_to_word = tokenize(corpus)
                a=list(word_to_ix.keys())
                a.sort()
                testword=st.selectbox("Chọn từ để kiểm tra",a)
        elif option_pre=="GloVe":
            selected_text = st.selectbox("Chọn bài để phân tích GloVe:", st.session_state.df2["text"])
            if  str(selected_text) !="" :
                doc = nlp(str(selected_text))
                sentences = list(doc.sents)
                # selected_sent=st.selectbox("Chọn câu xử lý CBOW",[str(sent) for sent in sentences])
                corpus=[str(sent) for sent in sentences]
                vocab, word_to_ix, ix_to_word = build_vocab(corpus)
                vocab_size = len(vocab)
                a=list(word_to_ix.keys())
                a.sort()
                testword=st.selectbox("Chọn từ để kiểm tra",a)
        elif option_pre=="BERT":
            selected_text = st.selectbox("Chọn bài để phân tích BERT:", st.session_state.df2["text"])
            if  str(selected_text) !="" :
                doc = nlp(str(selected_text))
                sentences = list(doc.sents)
        elif option_pre=="ELMo":
            selected_text = st.selectbox("Chọn bài để phân tích ELMo:", st.session_state.df2["text"])
            if  str(selected_text) !="" :
                doc = nlp(str(selected_text))
                sentences = list(doc.sents)




    if st.button("Xử lý biểu diễn"):
        st.session_state.df3 = st.session_state.df2.copy()

        

        if option_pre=="One-Hot Encoding":
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
        elif option_pre=="CBOW":

                # Tạo dữ liệu cho CBOW (window_size = 2)


            cbow_data = create_cbows(words, word_to_ix)

            # Chuyển đổi dữ liệu thành chỉ số
            def prepare_data(cbow_data, word_to_ix):
                inputs, targets = [], []
                for context, target in cbow_data:
                    inputs.append([word_to_ix[word] for word in context])
                    targets.append(word_to_ix[target])
                return torch.tensor(inputs), torch.tensor(targets)

            inputs, targets = prepare_data(cbow_data, word_to_ix)

            # Xây dựng mô hình CBOW
            class CBOWModel(nn.Module):
                def __init__(self, vocab_size, embed_dim):
                    super(CBOWModel, self).__init__()
                    self.embeddings = nn.Embedding(vocab_size, embed_dim)
                    self.linear = nn.Linear(embed_dim, vocab_size)

                def forward(self, context):
                    embeds = self.embeddings(context).mean(dim=1)  # Tính trung bình embedding
                    out = self.linear(embeds)
                    return out

            # Khởi tạo mô hình
            embed_dim = 10
            model = CBOWModel(len(vocab), embed_dim)
            loss_function = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)

            # Huấn luyện mô hình
            num_epochs = 100
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                output = model(inputs)
                loss = loss_function(output, targets)
                loss.backward()
                optimizer.step()
                if epoch % 10 == 0:
                    continue
                    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

            # Kiểm tra embedding của một từ

            
            word_idx = word_to_ix[testword]
            st.write(f"Embedding của '{testword}': {model.embeddings.weight[word_idx].detach().numpy()}")
            # print(f"Embedding của 'nlp': {model.embeddings.weight[word_idx].detach().numpy()}")
            
        elif option_pre=="skip gram":
            skipgram_data = create_skipgrams(words, word_to_ix)
            # skipgram_data = create_skipgrams(words, word_to_ix)

            # Chuyển đổi dữ liệu thành tensor
            def prepare_data(skipgram_data):
                inputs, targets = zip(*skipgram_data)
                return torch.tensor(inputs), torch.tensor(targets)

            inputs, targets = prepare_data(skipgram_data)

            # Xây dựng mô hình Skip-gram
            class SkipGramModel(nn.Module):
                def __init__(self, vocab_size, embed_dim):
                    super(SkipGramModel, self).__init__()
                    self.embeddings = nn.Embedding(vocab_size, embed_dim)
                    self.linear = nn.Linear(embed_dim, vocab_size)

                def forward(self, target):
                    embed = self.embeddings(target)
                    out = self.linear(embed)
                    return out

            # Khởi tạo mô hình
            embed_dim = 10
            model = SkipGramModel(len(vocab), embed_dim)
            loss_function = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)

            # Huấn luyện mô hình
            num_epochs = 100
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                output = model(inputs)
                loss = loss_function(output, targets)
                loss.backward()
                optimizer.step()
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            # result = st.text_area(label="ket qua", value=t)
            word_idx = word_to_ix[testword]
            st.write(f"Embedding của '{testword}': {model.embeddings.weight[word_idx].detach().numpy()}")
        elif option_pre=="GloVe":
            cooccurrence_matrix = build_cooccurrence_matrix(corpus, word_to_ix)
            glove_model = GloVe(vocab_size, embedding_dim=10)
            glove_model.train(cooccurrence_matrix, epochs=100)
            word_idx = word_to_ix[testword]
            st.write(f"Embedding của '{testword}': {glove_model.W[word_idx]}")
        elif option_pre=="BERT":
            model_name = "bert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            bow_rep=tokenizer([str(span) for span in sentences], padding=True, truncation=True, return_tensors="pt")
            df3 = pd.DataFrame(bow_rep["input_ids"].numpy(), index=sentences)
            st.dataframe(df3)
        elif option_pre=="ELMo":
            pass
            # elmo.signatures["default"](tf.constant([str(span) for span in sentences]))["elmo"]
            # df

elif st.session_state.step == 5:
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('wordnet')
    nlp = spacy.load("en_core_web_sm")
    st.header("Bước 5: Text Classification")

    option_mission =st.selectbox("Chọn nhiệm vụ:",
                                ["setiment analysis",
                                "topic classification",
                                "question answering",
                                ])
    if option_mission=="setiment analysis":
        option_dataset =st.selectbox("Chọn tập dữ liệu:", [
            "SST-2",
            "IMDb",
            "Yelp"])
        if option_dataset == "IMDb":
            st.image("imdb_result.png")
        elif option_dataset == "SST-2":
            st.image("sst2_result.png")
        elif option_dataset == "Yelp":
            pass
        
    if option_mission=="topic classification":
        option_dataset =st.selectbox("Chọn tập dữ liệu:", [
            "ag_news",
            # "dbpedia",
            # "yahoo_answers",
            # "yelp_review_full",
            # "yelp_review_polarity"
        ])
        st.image("ag_news_result.png")
        
        
    option_encode=st.selectbox("Chọn phương pháp vector hóa:",[
                                                        "TF-IDF",
                                                        "Bag of Words",
                                                        "One-Hot Encoding",
                                                        "Bag of n-grams",
                                                        
                                                        #    "Word2Vec",
                                                        #    "BERT"
                                                        ])
    option_model=st.selectbox("Chọn mô hình phân loại:", ["Logistic Regression", "SVM",
                                                        #   "Random Forest  ",
                                                        "Naive Bayes",
                                                        "knn",
                                                        "Decision Tree",
                                                        #   "Maximizing Likelihood", 
                                                        # "BERT"
                                                        ])
    if option_model=="Decision Tree":
        st.image("DecisionTree_result.png")
    percent_test_size = st.number_input("Nhập tỷ lệ dữ liệu test:", min_value=0.1, max_value=0.9, step=0.1, value=0.3)
    #
    # st.text_area("Nhập văn bản cần phân loại:")

    example=st.text_area("Nhập văn phân loại")    
    #     st.selectbox("Chọn bài viết để phân loại",st.session_state.df3["text"])
    if option_model=="Logistic Regression":
        pass
    if  st.button("Phân loại trên dữ liệu "):
        if option_model == "Logistic Regression":
            model_Classification = LogisticRegression(max_iter=1000)
        elif option_model == "SVM":
            model_Classification = SVC()

        elif option_model == "Naive Bayes":
            model_Classification = MultinomialNB()

        elif option_model == "knn":
            model_Classification = KNeighborsClassifier(n_neighbors=5)
        elif option_model == "Decision Tree":
            model_Classification = DecisionTreeClassifier()
            
            # --------------------------------------------------------------------
            # option_encode
            
        if option_encode == "Bag of N Words":
            model_encode = CountVectorizer()
        elif option_encode == "TF-IDF":
            model_encode = TfidfVectorizer()
        elif option_encode == "Bag of N Words":
            model_encode = CountVectorizer(ngram_range=(1,3))
        elif option_encode == "One-Hot Encoding":
            model_encode = CountVectorizer(binary=True)
        
        # elif option_encode == "BERT":

        #     model_encode = BertTokenizer.from_pretrained("bert-base-uncased")
            
            

        if option_mission=="setiment analysis":
            if option_dataset=="IMDb":
                dataset = load_dataset("imdb")
                train_texts = dataset["train"]["text"]
                train_labels = dataset["train"]["label"]
                test_texts = dataset["test"]["text"]
                test_labels = dataset["test"]["label"]
            elif option_dataset=="SST-2":
                dataset =  load_dataset("glue", "sst2")
                train_texts = dataset["train"]["sentence"]
                train_labels = dataset["train"]["label"]
                test_texts = dataset["validation"]["sentence"]
                test_labels = dataset["validation"]["label"]
                st.write("load data complete")
            elif option_dataset=="Yelp":
                dataset = load_dataset("yelp_polarity")
                train_texts = dataset["train"]["text"]
                train_labels = dataset["train"]["label"]
                test_texts = dataset["test"]["text"]
                test_labels = dataset["test"]["label"]
        elif option_mission=="topic classification":
            dataset = load_dataset("ag_news")
            train_texts = dataset["train"]["text"]
            train_labels = dataset["train"]["label"]
            test_texts = dataset["test"]["text"]
            test_labels = dataset["test"]["label"]
        elif option_mission=="question classification":
            dataset = load_dataset("CogComp/trec")
            train_texts = dataset["train"]["text"]
            train_labels = dataset["train"]["coarse_label"]
            test_texts = dataset["test"]["text"]
            test_labels = dataset["test"]["coarse_label"]
        # st.write("load data complete")
        all_texts=train_texts+test_texts
        all_labels=train_labels+test_labels
        train_texts, test_texts, train_labels, test_labels = train_test_split(
        all_texts, all_labels, test_size=0.2, random_state=42)
        pipline_cls=make_pipeline(model_encode,model_Classification)
        pipline_cls.fit(train_texts,train_labels)
        y_pred=pipline_cls.predict(test_texts)
        st.session_state.pipsave=pipline_cls
        
                # sentences.append({"title": row["title"], "sentence": sentence})
                
        
        st.write("Accuracy on test set: ",accuracy_score(test_labels,y_pred))

        if option_mission=="setiment analysis":

            
            df=st.session_state.df2
            df_title=[]
            df_sent=[]
            df_result=[]
            for idx, row in df.iterrows():
                for sentence in sent_tokenize(row["text"]):
                    df_title.append(row["title"])
                    df_sent.append(sentence)
                    df_result.append(predict_sentiment(str(sentence), pipline_cls))
            df_sent=pd.DataFrame({"title":df_title,"sentence":df_sent,"result":df_result})
            
            st.dataframe(df_sent)
            st.write(predict_sentiment(example, pipline_cls))
            
        elif option_mission=="topic classification":
            df=st.session_state.df2
            df_title=[]
            df_sent=[]
            df_result=[]
            for idx, row in df.iterrows():
                for sentence in sent_tokenize(row["text"]):
                    df_title.append(row["title"])
                    df_sent.append(sentence)
                    df_result.append(predict_topic(str(sentence), pipline_cls))
            df_sent=pd.DataFrame({"title":df_title,"sentence":df_sent,"result":df_result})
            
            st.dataframe(df_sent)
        elif option_mission=="question classification":
            df=st.session_state.df2
            df_title=[]
            df_sent=[]
            df_result=[]
            for idx, row in df.iterrows():
                for sentence in sent_tokenize(row["text"]):
                    df_title.append(row["title"])
                    df_sent.append(sentence)
                    df_result.append(predict_question(str(sentence), pipline_cls))
            df_sent=pd.DataFrame({"title":df_title,"sentence":df_sent,"result":df_result})
            
            st.dataframe(df_sent)
            
left, middle, right = st.columns(3, vertical_alignment="bottom")
print("step ",st.session_state.step)
if left.button("Quay lại"):
    # if st.session_state.step > 1:
    st.session_state.step -= 1
    st.rerun()
if right.button("Tiếp theo"):
    # if st.session_state.step <5:
    st.session_state.step = 1+st.session_state.step
    st.rerun()
if middle.button("Reset"):
    st.session_state.step = 1
    st.rerun()
