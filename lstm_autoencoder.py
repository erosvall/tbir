
# Originally adapted from https://github.com/keras-team/keras/issues/1401

# Requires Keras and Tensorflow backend


#from keras.layers import containers
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, RepeatVector, Input
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.layers.recurrent import LSTM
from keras.utils import plot_model

corpus = open("qa.894.raw.train.txt").read().lower().splitlines()

## Extracting Training data and initializing some variables for the model
train_x = corpus[0::2] # extract every second item from the list
train_t = corpus[1::2]
N = len(train_x)
latent_dimension = 10
sequence = 30

## Preprocessing of the words 
token = Tokenizer(num_words=None)
token.fit_on_texts(corpus)
train_x = token.texts_to_sequences(train_x)

train_x = np.array(train_x)
train_x = pad_sequences(train_x)

map(np.array,train_x)

train_x = train_x.reshape((1,N,sequence))

print('step 1')

## Build and train Autoencoder
#autoencoder = Sequential()
#encoder = LSTM(latent_dimension, input_shape = (N,sequence) , return_sequences=True)
#decoder = LSTM(sequence, input_shape = (N, latent_dimension) , return_sequences=True)
#autoencoder.add(encoder)
#autoencoder.add(decoder)
#autoencoder.compile(loss='categorical_crossentropy', optimizer='RMSprop')
#autoencoder.fit(train_x,train_x, epochs = 1)

print('step 2')
inputs = Input(shape = (sequence,N))
encoder = LSTM(latent_dimension)(inputs)
print('step3')
decoder = RepeatVector(sequence)(encoder)
decoder = LSTM(N, return_sequences = True)(decoder)
print('step3.4')
autoencoder = Model(inputs, decoder)
autoencoder.compile(loss='categorical_crossentropy', optimizer='RMSprop')
print('step 4')
autoencoder.fit(train_x,train_x, epochs = 1)

print('step 5')


plot_model(autoencoder, to_file='model.png')
a = autoencoder.predict(train_x)
print(a)

print(train_x)
print(a.shape)

#encoder_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(encoder).output)