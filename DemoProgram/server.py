# -*- coding: utf-8 -*-
from flask import Flask, render_template, request

import datetime
import tensorflow as tf
import numpy as np
import os
from pickle import load
from numpy import argmax
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import cv2

path = os.path.join(os.getcwd(),'static/img/user_img')
if not os.path.exists(path):
    os.mkdir(path)

# 나머지 선택지에 대해서 저장할 변수들...?
# with open('log.txt','wt') as f: # file open write txt
#     f.write('start')


app = Flask(__name__)

@app.route("/")
def index():
	return render_template('index.html')

@app.route("/Test")
def test():
	return render_template('Test.html')


@app.route("/Test/House")
def house():
	return render_template('House.html')

@app.route("/Test/Tree")
def tree():
	return render_template('Tree.html')

@app.route("/Test/Person")
def person():
	return render_template('Person.html')


@app.route("/Test/Result", methods=['GET','POST'])
def result():
	# H,T,P 어느 result인지 순서도 넘겨줘서 H,T,P 순서대로 Test를 할 수 있도록
	HTP = None

	if request.method == 'POST':
		HTP = request.form['test_name']
		test_path = os.path.join(path,HTP);
		f = request.files['file']
		cap_img = os.path.join(test_path,f.filename)
		f.save(cap_img) # user_img경로에 유저가 올린 이미지 저장

	print("img path : ",cap_img)
	# load the tokenizer
	print("load tokenizer ...")
	tokenizer = load(open('tokenizer.pkl', 'rb')) # ----
	print("tokenizer loading succeess")
	# pre-define the max sequence length (from training)
	max_length = 16
	# load the model
	print("load model ...")
	model = load_model('model-ep006-loss0.469-val_loss0.487-anchor.h5') # ----
	print("model loading success")
	# load and prepare the photograph
	photo = extract_features(cap_img) # -- 이부분들을 페이지 랜더링 후에 보여주면 안되나..?
	# generate description
	print("generate description")
	description = generate_desc(model, tokenizer, photo, max_length)
	print("done!")

	print("description : ",description)
	print("file : ",f.filename)
	# img= cv2.imread('example.png', cv2.IMREAD_COLOR)
	# cv2.imshow('test',img)
	# cv2.waitKey(0)
	return render_template('Result.html', image_path=f.filename, result_title = HTP, description=description)


# extract features from each photo in the directory
def extract_features(filename):
    # load the model
    model = InceptionV3() # ----
    #model = load_model('sketch_inceptionv3.h5')
    # re-structure the model
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output) # ----
    # load the photo
    image = load_img(filename, target_size=(299, 299))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    print("Predict...")
    # get features
    feature = model.predict(image, verbose=0)
    print("succeess")
    return feature


# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]  # ----
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)  # ----
        # predict next word
        yhat = model.predict([photo, sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text


# @app.before_first_request
# def befor_first_request():
# 	print("앱이 기동되고 나서 첫번쨰 HTTP요청에만 응답합니다.")

# @app.before_request
# def before_request():
# 	print("매 HTTP요청이 처리되기 전에 실행됩니다.v")

# @app.teardown_request
# def teardown_request(exception):
# 	print("매 HTTP 요청의 결과가 브러우저에 응답하고 나서 호출됩니다.v")

# @app.teardown_appcontext
# def teardown_appcontexxt(exception):
# 	print("HTTP 요청의 애플리케이션 컨텍스트가 종료될 때 실행됩니다.v")

# @app.after_request
# def after_request(response):
# 	print("매 HTTP요청이 처리되고 나서 실행됩니다.")

if __name__ == '__main__':



   app.run(debug = True)