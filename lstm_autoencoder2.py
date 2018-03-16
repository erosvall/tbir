
# Originally adapted from https://github.com/keras-team/keras/issues/1401
# https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
# Requires Keras and Tensorflow backend


#from keras.layers import containers
from keras.models import Sequential, Model,load_model
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model,to_categorical,print_summary
import numpy as np
import os.path

def preprocess(text,corpus):
	token = Tokenizer(num_words=None)
	token.fit_on_texts(corpus)
	text = token.texts_to_sequences(text)

	text = np.array(text)
	text = pad_sequences(text)

	map(np.array,text)

	text = to_categorical(text)
	(N,sequence,voc) = text.shape

	return text, N, sequence, voc

def load_dataset(filename,k = 0):
	corpus = open(filename).read().lower().splitlines()

	## Extracting Training data and initializing some variables for the model
	x = corpus[0:2*k-1:2] # extract every second item from the list
	t = corpus[1:2*k-1:2]
	x, N, sequence, voc = preprocess(x,corpus)
	t = preprocess(t,corpus)

	return x,t,N,sequence,voc

def build_autoencoder(latent_dimension1,latent_dimension2,voc):
	autoencoder = Sequential()
	encoder = LSTM(latent_dimension1,input_shape = (None,voc), return_sequences=True)
	encoder2 = LSTM(latent_dimension2,return_sequences=True)
	decoder = LSTM( voc, return_sequences=True)
	autoencoder.add(encoder)
	autoencoder.add(encoder2)
	autoencoder.add(decoder)
	autoencoder.add(Dense(voc, activation='softmax'))
	autoencoder.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics = ['acc'])
	return autoencoder



#Dimensionality reduction in encoder1 and encoder 2
latent_dimension1 = 140 
latent_dimension2 = 50
epochs = 2
load_data = True
file_id = 'Autoencoder_' +str(epochs)+'_'+ str(latent_dimension1) + '_' + str(latent_dimension2) +'.h5'

if load_data:
	# This function fetches the dataset from the file and fills both X and T with k number of datapoints
	train_x, train_t,N,sequence,voc = load_dataset("qa.894.raw.train.txt",100)

## Build and train Autoencoder
if os.path.exists(file_id):
	print('\nModel with these parameters found, loading model\n')
	autoencoder = load_model(file_id)
else:
	print('\nNo model with these parameters was found, building new model.\n')
	autoencoder = build_autoencoder(latent_dimension1,latent_dimension2,voc)
	early_stopping = EarlyStopping(monitor='val_loss', patience=4)
	autoencoder.fit(train_x,train_x, epochs = epochs, validation_split=0.2,callbacks=[early_stopping])
	autoencoder.save(file_id)

print('Autoencoder parameters')
print_summary(autoencoder)
#encoder_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(encoder).output)