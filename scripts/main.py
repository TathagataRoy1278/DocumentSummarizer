

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import requests
from bs4 import BeautifulSoup

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

import streamlit as st

from copy import deepcopy

stop_words = stopwords.words('english')
total_length = 0
link = st.text_input("Enter link to article : ","https://medium.com/sciforce/towards-automatic-text-summarization-extractive-methods-e8439cd54715")

word_embeddings = {}
@st.cache
def load_word_embeddings():
    f = open("/media/tatan/A0A04E3AA04E16E6/word_vectors/glove.6B.100d.txt",encoding = 'utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()



def sentencevector_avg(word_vectors):
    avg_vector = []
    if len(word_vectors)!=0:
        avg_vector = np.array([sum([vector[axis] for vector in word_vectors])/len(word_vectors) for axis in range(len(word_vectors[0]))])
    else:
        avg_vector = np.zeros((100,))
    return avg_vector


def sentencevector_avg_weighted(word_vectors,tmp_vectors,weight):
    for i in range(len(tmp_vectors)):
        for j in range(len(tmp_vectors[0])):
            tmp_vectors[i][j] = weight*tmp_vectors[i][j]
    word_vectors+=tmp_vectors
    avg_vector = []
    if len(word_vectors)!=0:
        avg_vector = np.array([sum([vector[axis] for vector in word_vectors])/len(word_vectors) for axis in range(len(word_vectors[0]))])
    else:
        avg_vector = np.zeros((100,))
    return avg_vector



def remove_stopwords_and_punctuations(sen):
    tokenizer = RegexpTokenizer(r'\w+')
    sen = " ".join(tokenizer.tokenize(sen))
    new_vec = " ".join([word for word in sen.split() if word not in stop_words])
    return new_vec



def get_word_vectors(sentence):
    sentence = remove_stopwords_and_punctuations(sentence)
    word_vectors = []
    for word in sentence.split():
        try:
            word_vectors.append(deepcopy(word_embeddings[word.lower()]))
        except KeyError as e:
            print(e)
    return word_vectors




def getSummary(title,sentence_tokens):
    summary = []
    title_vector = sentencevector_avg(get_word_vectors(title))
    for sentence_token in sentence_tokens:
        sentence_vector = []

        sentence_vector = sentencevector_avg(get_word_vectors(sentence_token))

        tmp = pearsonr(sentence_vector,title_vector)[0]
        summary.append((tmp,sentence_token))
        title_vector = sentencevector_avg_weighted(get_word_vectors(title),get_word_vectors(sentence_token),0.08)

    return summary






st.title("Document Summarizer")




data_load_state = st.text('Loading data...')

html = requests.get(link).text
soup = BeautifulSoup(html,'html.parser')
load_word_embeddings()

title = soup.find('title').text
paras = soup.findAll('p')
text = [i.text for i in paras if len(i.text)>70]
text = "".join(text)
sentence_tokens = sent_tokenize(text)
total_length = len(sentence_tokens)

data_load_state.text("Data Loaded")


recquired_length = float(st.text_input("Enter the desired length of the summary wrt the text, eg - 0.25 means 25% of the total length",0.33333))


st.subheader("Title")
st.write(title)

st.subheader("Main Body Text")
st.write(text)

st.subheader("Summary")
summary = [(i,s) for i,s in enumerate(getSummary(title,sentence_tokens))]
summary.sort(key = lambda x: x[1][0], reverse = True)

final_summary = sorted(summary[:int(recquired_length*total_length)],key = lambda x: x[0])

summary = ""
for i in final_summary:
    summary+=i[1][1]+"\n"

st.write(summary)
st.write("The length of the summary is "+str(len(summary))+" which is "+str(float(len(summary))/len(text)*100)+"% of the originl length")

