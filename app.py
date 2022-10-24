'''
	Contoh Deloyment untuk Domain Natural Language Processing (NLP)
	Orbit Future Academy - AI Mastery - KM Batch 3
	Tim Deployment
	2022
'''

# =[Modules dan Packages]========================

from flask import Flask,render_template,request,jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from joblib import load
import re
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from transformers import BertTokenizer, AutoTokenizer, AutoModel
from transformers import TFBertForSequenceClassification

from fungsi import *

# =[Variabel Global]=============================

app   = Flask(__name__, static_url_path='/static')
model = None

# stopwords_ind = None
# key_norm      = None
# factory       = None
# stemmer       = None
# vocab         = None

# =[Routing]=====================================

# [Routing untuk Halaman Utama atau Home]	
@app.route("/")
def beranda():
    return render_template('index.html')

# [Routing untuk API]		
@app.route("/api/deteksi",methods=['POST'])
def apiDeteksi():
	# Nilai default untuk string input 
	text_input = ""
	
	if request.method=='POST':
		# Set nilai string input dari pengguna
		text_input = request.form['data']
		
		bert_tokenizer = AutoTokenizer.from_pretrained('indolem/indobertweet-base-uncased')

		# Text Pre-Processing
		text_input = casefolding(text_input)

		# TF-IDF
		input_text_tokenized = bert_tokenizer.encode(text_input,
                                             truncation=True,
                                             padding='max_length',
                                             return_tensors='tf')

		# Prediksi (Penipuan, Promo, atau Normal)
		hasil = model(input_text_tokenized)          # Lakukan prediksi
		bert_output = tf.nn.softmax(hasil[0], axis=-1) 
		labels = ['Angry', 'Fear', 'Happiness', 'Love', 'Sadness']

		label = tf.argmax(bert_output, axis=1)
		label = label.numpy()

		hasil_prediksi = labels[label[0]]
		
		# Return hasil prediksi dengan format JSON
		return jsonify({
			"data": hasil_prediksi,
		})

# =[Main]========================================

if __name__ == '__main__':
	
	# # Setup
	# stopwords_ind = stopwords.words('indonesian')
	# stopwords_ind = stopwords_ind + more_stopword
	
	# key_norm = pd.read_csv('key_norm.csv')
	# key_norm2 = pd.read_excel('kamus_singkatan.xlsx')
	
	# factory = StemmerFactory()
	# stemmer = factory.create_stemmer()
	
	# vocab = pickle.load(open('kbest_feature.pickle', 'rb'))
	
	# Load model yang telah ditraining

	model = TFBertForSequenceClassification.from_pretrained('indolem/indobertweet-base-uncased', num_labels=5, from_pt=True)
	model.load_weights('bert-model.h5')

	# Run Flask di localhost 
	app.run(host="localhost", port=5000, debug=True)
	
	


