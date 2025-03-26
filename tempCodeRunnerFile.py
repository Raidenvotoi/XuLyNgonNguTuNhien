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


import praw
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nlp = spacy.load("en_core_web_sm")
# import svgling # Import svgling f