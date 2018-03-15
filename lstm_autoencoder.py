
# Originally adapted from https://github.com/keras-team/keras/issues/1401

# Requires Keras and Tensorflow backend


#from keras.layers import containers
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.layers.recurrent import LSTM

corpus = open("qa.894.raw.train.txt").read().lower().splitlines()

## Extracting Training data and initializing some variables for the model
train_x = corpus[0::2] # extract every second item from the list
train_t = corpus[1::2]
N = len(train_x)
dim = 10
padding = 30

## Preprocessing of the words 
token = Tokenizer(num_words=None)
token.fit_on_texts(corpus)
train_x = token.texts_to_sequences(train_x)

train_x = np.array(train_x)
train_x = pad_sequences(train_x)

map(np.array,train_x)

train_x = train_x.reshape((1,N,padding))


## Build and train Autoencoder
autoencoder = Sequential()
encoder = LSTM(dim, input_shape = (N,padding) , return_sequences=True)
decoder = LSTM(padding, input_shape = (N, dim), return_sequences=True)
autoencoder.add(encoder)
autoencoder.add(decoder)
autoencoder.compile(loss='categorical_crossentropy', optimizer='RMSprop')
autoencoder.fit(train_x,train_x, epochs = 10)

print(autoencoder.predict(train_x))
print(train_x)

#encoder_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(encoder).output)