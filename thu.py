from multiprocessing import Process, Manager
import os
import time
from datasets import load_dataset, load_from_disk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

# Định nghĩa danh sách model theo dạng dictionary để có thể khởi tạo lại
model_encoders = {
    "CountVectorizer": CountVectorizer,
    "TfidfVectorizer": TfidfVectorizer,
    "CountVectorizer (ngram)": lambda: CountVectorizer(ngram_range=(1, 2)),
    "CountVectorizer (binary)": lambda: CountVectorizer(binary=True)
}

model_classifiers = {
    "DecisionTree": DecisionTreeClassifier,
    "RandomForest": RandomForestClassifier,
    "MultinomialNB": MultinomialNB,
    "LogisticRegression": lambda: LogisticRegression(max_iter=500),
    "KNN": KNeighborsClassifier
}

def worker(encoder_name, classifier_name, return_dict, key):
    # Mỗi tiến trình phải tự load lại dataset
    dataset = load_from_disk("ag_news_combined")
    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    # Khởi tạo mô hình
    encoder = model_encoders[encoder_name]()
    classifier = model_classifiers[classifier_name]()

    # Huấn luyện pipeline
    pipeline_cls = make_pipeline(encoder, classifier)
    pipeline_cls.fit(train_texts, train_labels)
    accuracy = pipeline_cls.score(test_texts, test_labels)
    
    print(f"Accuracy: {accuracy:.4f} using {encoder_name} and {classifier_name}")
    return_dict[key] = accuracy

if __name__ == "__main__":
    # Load dataset và lưu để các process có thể đọc lại


    manager = Manager()
    return_dict = manager.dict()
    processes = []

    # Tạo các tiến trình
    for encoder_name in model_encoders:
        for classifier_name in model_classifiers:
            key = f"{classifier_name} with {encoder_name}"
            p = Process(target=worker, args=(encoder_name, classifier_name, return_dict, key))
            p.start()
            processes.append(p)

    # Đợi tất cả tiến trình hoàn thành
    for p in processes:
        p.join()

    # Vẽ biểu đồ kết quả
    results = dict(return_dict)
    labels, accuracies = zip(*results.items())

    plt.figure(figsize=(10, 8))
    plt.barh(labels, accuracies, color='skyblue')
    plt.xlabel('Accuracy')
    plt.title('Model Accuracy with Different Encoders and Classifiers')
    plt.show()
