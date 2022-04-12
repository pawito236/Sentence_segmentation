# -*- coding: utf-8 -*-

import glob
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import re
import pycrfsuite

from flask import Flask, escape, request, jsonify, render_template
from markupsafe import escape
from werkzeug.utils import secure_filename
import requests
import time

app = Flask(__name__,static_url_path='/static')
app.config["UPLOAD_FOLDER"] = "static/"

@app.route('/')
def index():
   return 'Index Page'

@app.route('/mysite')
def indexweb():
   return render_template('myweb.html')


enders = ["ครับ","ค่ะ","คะ","นะคะ","นะ","จ้ะ","จ้า","จ๋า","ฮะ", #ending honorifics
          #enders
          "ๆ","ได้","แล้ว","ด้วย","เลย","มาก","น้อย","กัน","เช่นกัน","เท่านั้น",
          "อยู่","ลง","ขึ้น","มา","ไป","ไว้","เอง","อีก","ใหม่","จริงๆ",
          "บ้าง","หมด","ทีเดียว","เดียว",
          #demonstratives
          "นั้น","นี้","เหล่านี้","เหล่านั้น",
          #questions
          "อย่างไร","ยังไง","หรือไม่","มั้ย","ไหน","อะไร","ทำไม","เมื่อไหร่"]
starters = ["ผม","ฉัน","ดิฉัน","ชั้น","คุณ","มัน","เขา","เค้า",
            "เธอ","เรา","พวกเรา","พวกเขา", #pronouns
            #connectors
            "และ","หรือ","แต่","เมื่อ","ถ้า","ใน",
            "ด้วย","เพราะ","เนื่องจาก","ซึ่ง","ไม่",
            "ตอนนี้","ทีนี้","ดังนั้น","เพราะฉะนั้น","ฉะนั้น",
            "ตั้งแต่","ในที่สุด",
            #demonstratives
            "นั้น","นี้","เหล่านี้","เหล่านั้น"]

def extract_features(doc, window=2, max_n_gram=3):
    doc_features = []
    #paddings for word and POS
    doc = ['xxpad' for i in range(window)] + doc + ['xxpad' for i in range(window)]
    doc_ender = []
    doc_starter = []
    #add enders/starters
    for i in range(len(doc)):
        if doc[i] in enders:
            doc_ender.append('ender')
        elif doc[i] in starters:
            doc_starter.append('starter')
        else:
            doc_ender.append('normal')
            
    #for each word
    for i in tqdm(range(window, len(doc)-window)):
        #bias term
        word_features = ['bias']
        
        #ngram features
        for n_gram in range(1, min(max_n_gram+1,2+window*2)):
            for j in range(i-window,i+window+2-n_gram):
                feature_position = f'{n_gram}_{j-i}_{j-i+n_gram}'
                
                word_ = f'{"|".join(doc[j:(j+n_gram)])}'
                word_features += [f'word_{feature_position}={word_}']
                
                ender_ =  f'{"|".join(doc_ender[j:(j+n_gram)])}'
                word_features += [f'ender_{feature_position}={ender_}']
                
                starter_ =  f'{"|".join(doc_starter[j:(j+n_gram)])}'
                word_features += [f'starter_{feature_position}={starter_}']
        
        #append to feature per word
        doc_features.append(word_features)
    return doc_features


@app.route('/predict', methods = ['GET', 'POST'])
def myprediction():
    words = ""
    if request.method == 'POST':
      words = str(request.get_data())
      #words = request.files['text']
      #url = request.form['text']
      #words = requests.get(url)
    list_words = words.split(" ")
    #try
    docf = extract_features(list_words, window = 2, max_n_gram = 3)
    #print(docf)
    # Predict (using test set)

    tagger = pycrfsuite.Tagger()
    tagger.open('sub1-crf-Final.model')
    y_pred = tagger.tag(docf)
    #print(y_pred)

    df = pd.DataFrame({'Predicted':y_pred,'word':words})

    df.Predicted[len(df['word'])-1] = "I_SENT"

    # Find index of "O"
    index_sub3_EOB=df[(df['Predicted']=='O')].index

    #fill E,B_SENT
    df.loc[index_sub3_EOB-1, "Predicted"] = "E_SENT"
    df.loc[index_sub3_EOB[:-1]+1, "Predicted"] = "B_SENT"

    df.Predicted[0] = "B_SENT"
    df.Predicted[len(df['word'])-1] = "E_SENT"

    df['Id'] = [i for i in range(len(df['word']))]

    df_Mr_Id=df[(df['word']=='น.ส.')|
                        (df['word']=='นาย')|
                        (df['word']=='นาง')|
                        (df['word']=='นางสาว')]['Id']

    df_Mr_Id=pd.DataFrame(df_Mr_Id)

    list_Mr_Id=list(df_Mr_Id['Id'])
    Mr_id=[]
    Mr_word=[]
    li = []
    for id in list_Mr_Id:
        add_id=id+1
        if df['Predicted'][add_id]=='O':
            # Mr_id.append(range(id,id+3))
            print('***',range(id,id+3))
            # print()
            print(df['word'][id-1:id+3],df['Predicted'][id-2:id+3])

    df_Said_Id=df[(df['word']=='กล่าว')]['Id']

    df_Said_Id=pd.DataFrame(df_Said_Id)

    list_Said_Id=list(df_Said_Id['Id'])
    i=0
    for id in list_Said_Id:
        # print('id',id)
        if df['word'][id]==" ":
            if df['Predicted'][id]!='O':
                df.loc[id, "Predicted"] = "O"
                df.loc[id-1, "Predicted"] = "E_SENT"
                df.loc[id+1, "Predicted"] = "B_SENT"

    print(df)



    return str(df.Predicted)

app.run(host='0.0.0.0',port=3000,debug=True)

