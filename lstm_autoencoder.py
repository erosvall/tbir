
# Originally adapted from https://github.com/keras-team/keras/issues/1401

#from keras.layers import containers
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import *
import numpy as np
from keras.layers.recurrent import LSTM

corpus = open("qa.894.raw.train.txt").read().lower().splitlines()


train_x = corpus[0::2] # extract every second item from the list
train_t = corpus[1::2]
N = len(train_x)
dim = 10
padding = 30

token = Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ", char_level=False, oov_token=None)
token.fit_on_texts(corpus)
train_x = token.texts_to_sequences(train_x)

train_x = np.array(train_x)
train_x = pad_sequences(train_x,maxlen = padding,dtype = "int32",padding="pre",truncating = "pre", value = 0.0)

map(np.array,train_x)


train_x = train_x.reshape((1,N,padding))
#for i in range(0,N):
#	train_x[i] = np.array(train_x[i])

autoencoder = Sequential()
encoder = LSTM(output_dim=dim, input_shape = [N,padding], activation='tanh' , return_sequences=True)
decoder = LSTM(output_dim=padding, input_shape = [N, dim], activation='tanh', return_sequences=True)
autoencoder.add(encoder)
autoencoder.add(decoder)
autoencoder.compile(loss='mse', optimizer='RMSprop')
autoencoder.fit(train_x,train_x, epochs = 10)

#encoder_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(encoder).output)