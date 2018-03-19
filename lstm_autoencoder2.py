
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

def build_autoencoder(latent_dimension1,latent_dimension2,voc,x,e):
	autoencoder = Sequential()
	autoencoder.add(LSTM(latent_dimension1,input_shape = (None,voc), return_sequences=True))
	autoencoder.add(LSTM(latent_dimension2,return_sequences=True))
	autoencoder.add(LSTM( voc, return_sequences=True))
	autoencoder.add(Dense(voc, activation='softmax'))
	autoencoder.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics = ['acc'])
	early_stopping = EarlyStopping(monitor='val_loss', patience=4)
	autoencoder.fit(x,x, epochs = e, validation_split=0.2,callbacks=[early_stopping])
	return autoencoder

def build_classifier(source_model,voc,x,t,e,l1,l2):
	classifier = Sequential()
	classifier.add(LSTM(l1,input_shape=(None,voc),return_sequences=True))
	classifier.add(LSTM(l2,return_sequences=True))
	classifier.layers[0].set_weights(source_model.layers[0].get_weights())
	classifier.layers[1].set_weights(source_model.layers[1].get_weights())
	classifier.add(Dense(voc,activation='softmax'))
	classifier.summary()
	classifier.compile(loss='categorical_crossentropy',optimizer='Adam',metrics = ['acc'])
	early_stopping = EarlyStopping(monitor='val_loss', patience=4)
	classifier.fit(x,t, epochs = e,validation_split = 0.2, callbacks = [early_stopping])
	# We should look at fine tuning as well, basically evaluate  on train_t
	return classifier


#Dimensionality reduction in encoder1 and encoder 2
latent_dimension1 = 140 
latent_dimension2 = 50
epochs = 2
load_data = True
file_id = 'Autoencoder_' +str(epochs)+'_'+ str(latent_dimension1) + '_' + str(latent_dimension2) +'.h5'


if load_data:
	# This function fetches the dataset from the file and fills both X and T with k number of datapoints
	train_x, train_t,N,sequence,voc = load_dataset("qa.894.raw.train.txt",300)


## Build and train Autoencoder
if os.path.exists(file_id):
	print('\n Model with these parameters found, loading model \n')
	autoencoder = load_model(file_id)
else:
	print('\n No model with these parameters was found, building new model.\n')
	autoencoder = build_autoencoder(latent_dimension1,latent_dimension2,voc,train_x,epochs)
	autoencoder.save(file_id)
print('Autoencoder parameters')
autoencoder.summary()


classifier = build_classifier(autoencoder,voc,train_x,train_t,epochs,latent_dimension1,latent_dimension2)
print(test_t.shape)
print(test_x.shape)
test_x, test_t,N_t,sequence_t,voc_t = load_dataset("qa.894.raw.test.txt",300)
classifier.evaluate(test_x,test_t)