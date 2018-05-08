from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Embedding, Input, Dropout, concatenate, RepeatVector
from keras.layers.recurrent import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model
from keras.backend import argmax
from keras.callbacks import ModelCheckpoint
from tensorflow import InteractiveSession
from keras import regularizers
from nltk.corpus import wordnet as wn
from tqdm import tqdm
import numpy as np
import os.path
import argparse
import sys
import time

def load_cnn(filename):
    # Returns a matrix where each row corresponds to imageXXXX, 
    # note that index determines image number
    images = open(filename).read().splitlines()
    for i,img in enumerate(images):
        images[i] = img.split(',')
        images[i][0] = int(images[i][0][5:])
    images = np.asarray(images).astype(float)
    images = images[images[:,0].argsort()]
    return images[:,1:]

def match_img_features(questions,img_features):
    return np.asarray(list(map(lambda x: 
                img_features[int(x.split('image')[-1].split(' ')[0])-1],
                questions)))
                
def preprocess(text, token):
    text = token.texts_to_sequences(text)
    text = pad_sequences(
        text,
        maxlen = 30)
    # text = to_categorical(text, len(token.word_index.items())+1)
    (N, sequence) = text.shape
    voc = len(token.word_index.items())+1
    return text, N, sequence, voc

def q_preprocess(text, token):  
    text = token.texts_to_sequences(text)
    text = pad_sequences(text,maxlen = 30)
    # text = to_categorical(text, len(token.word_index.items())+1)
    (N, sequence) = text.shape
    voc = len(token.word_index.items())+1
    return text, N, sequence, voc

def a_preprocess(text, token):
    text = token.texts_to_sequences(text)
    text = pad_sequences(text,maxlen = 11,padding='post')
    pre_cat_text = text
    text = to_categorical(text, len(token.word_index.items())+1)
    (N, sequence,voc) = text.shape
    return text, N, sequence, voc, pre_cat_text

def multiple_hot(sequence):
    sum = sequence[0]
    for i in range(1,len(sequence)):
            sum += sequence[i]
    sum[0] = 0
    return sum

def load_dataset(filename, k=0, token=None,img_filename=None):  
    corpus = open(filename).read().lower().splitlines()
    if not img_filename is None:
        img_features = load_cnn(img_filename)
        questions = corpus[0:2*k:2]
        imgs = match_img_features(questions,img_features)
    if token is None:
        token = Tokenizer(oov_token='~')#,filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~')
        token.fit_on_texts(corpus)
    q_corpus, N, sequence, voc = q_preprocess(corpus, token)
    a_corpus, _, _, _, pre_cat_a_corpus = a_preprocess(corpus, token)
    # Extracting Training data and initializing some variables for the model
    x = q_corpus[0:2*k:2]  # extract every second item from the list
    t = a_corpus[1:2*k:2]
    #t = np.asarray(list(map(multiple_hot,t)))
    return x, imgs,t, N, sequence, voc, token