import random
import requests
import telebot
from telebot import types
import linecache
import urllib
import numpy as np
import pandas as pd
from nltk.tokenize import (
    sent_tokenize,
    word_tokenize,
    TweetTokenizer,
    WordPunctTokenizer,
    WhitespaceTokenizer,
    LegalitySyllableTokenizer,
    SyllableTokenizer,
)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report
import os
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
from tqdm.notebook import tqdm
import unicodedata
with zipfile.ZipFile('Bookclass.zip', 'r') as z: z.extractall()
with zipfile.ZipFile('data.zip', 'r') as z: z.extractall()
def func(path):
    with open(path, encoding='utf-8') as f:
        contents = f.readlines()
        a,b=int(contents[0].split(',')[1][2]),int(contents[0].split(',')[2][0])
    for i in range(len(contents)):
        contents[i]=contents[i].split(',')[0]
    df=pd.DataFrame([contents]).T
    df[1]=[a]*df.shape[0]
    df[2]=[b]*df.shape[0]
    return df
tmp=os.listdir('data')
df=pd.DataFrame([])
for i in tmp:
    for j in os.listdir(u'data/'+i):
        if j != u'.ipynb_checkpoints':
            df=pd.concat([df,func(u'data/'+i+'/'+j)],axis=0,ignore_index=True)
df=df.sample(frac=1)
xtraining_data=df[0].iloc[0:600].reset_index(drop=True)
xtest_data=df[0].iloc[600:-1].reset_index(drop=True)
def classer(x):
    len = x.shape[0]
    classes=8
    a=pd.DataFrame([])
    for i in x:
        c=i
        tmp=pd.DataFrame(np.zeros(8)).T
        tmp[c]=1
        a=pd.concat([a,tmp],axis=0,ignore_index=True)
    return a.reset_index(drop=True)
ytraining_data1=df[1].iloc[0:600].reset_index(drop=True)
ytest_data1=df[1].iloc[600:-1].reset_index(drop=True)
ytraining_data2=df[2].iloc[0:600].reset_index(drop=True)
ytest_data2=df[2].iloc[600:-1].reset_index(drop=True)
ytraining_data1=classer(ytraining_data1)
ytraining_data2=classer(ytraining_data2)
ytest_data1=classer(ytest_data1)
ytest_data2=classer(ytest_data2)
def Token(xtr):
    tokenizer=Tokenizer(num_words=1000,oov_token='<OOV>')
    tokenizer.fit_on_texts(df[0])
    word_index=tokenizer.word_index
    training_seq=tokenizer.texts_to_sequences(xtr)
    training_padded=pad_sequences(training_seq,maxlen=30,padding='post',truncating='post')
    return training_padded
test_padded=Token(xtest_data)
training_padded=Token(xtraining_data)
model_lang = tf.keras.Sequential([
    tf.keras.layers.Embedding(1000, 30, input_length=30),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='sigmoid'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(8, activation='softmax')
])
model_lang.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
history = model_lang.fit(training_padded, ytraining_data1, epochs=60, validation_data=(test_padded, ytest_data1), verbose=2,batch_size=8)
model_book = tf.keras.Sequential([
    tf.keras.layers.Embedding(1000, 30, input_length=30),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='sigmoid'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(8, activation='softmax')
])
model_book.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
history2 = model_book.fit(training_padded, ytraining_data2, epochs=60, validation_data=(test_padded, ytest_data2), verbose=2,batch_size=8)
tmp = pd.DataFrame([])
for j in range(1, 8):
    with open('Bookclass/Class book'+str(j)+'.txt', encoding='utf-8') as f:
        contents = f.readlines()
    bookdf=pd.DataFrame([contents]).T
    for i in range(len(contents)):
        bookdf.loc[i, 1]=contents[i][-5]
        bookdf.loc[i, 2]=contents[i][-3]
        bookdf.loc[i, 0]=contents[i][:-7]
    tmp=pd.concat([tmp,bookdf],axis=0,ignore_index=True)
bookdf = tmp
bookdf
def getstats(text):
    text=pd.DataFrame([text])
    text=text[0]
    predicted_categories = model_book.predict(Token(text))
    predicted_categories2 = model_lang.predict(Token(text))
    lvldf = pd.DataFrame(predicted_categories.T)
    lngdf = pd.DataFrame(predicted_categories2.T)
    max_index = lvldf[0].idxmax()
    max_index2 = lngdf[0].idxmax()
    if max_index == 0 and max_index2 == 0:
        return [-1, -1]
    return max_index2, max_index
def getbook(max_index2, max_index):
    tmpdf = []
    for i in range(len(bookdf)):
        if bookdf[1][i] == str(max_index2) and bookdf[2][i] == str(max_index):
            tmpdf.append(bookdf[0][i])
        if max_index == 0 and bookdf[1][i] == str(max_index2):
            tmpdf.append(bookdf[0][i])
        if max_index2 == 0 and bookdf[2][i] == str(max_index):
            tmpdf.append(bookdf[0][i])
        if max_index == -1 and max_index2 == -1:
            return "Язык отсутствует в базе данных или неверный ввод"
    return random.choice(tmpdf)
TOKEN = '6694685357:AAHxLH1pdqKvxu_WnfcnMFmeJGzw7M5d6Ps'
bot = telebot.TeleBot(TOKEN)
@bot.message_handler(commands=['start'])
def start(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard = True)
    bot.send_message(message.chat.id, 'Привет, {0.first_name}, это ITBot, он поможет вам найти книги для узучения языков программирования.'.format(message.from_user), reply_markup = markup)
@bot.message_handler(content_types=['text'])
def bot_message(message):
    msgdata = getstats(message.text)
    bot.send_message(message.chat.id, 'Вот книга для изучения языка по вашему запросу: '+ getbook(msgdata[0], msgdata[1]), disable_web_page_preview=True)
bot.polling(none_stop = True)
