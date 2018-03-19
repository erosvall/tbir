# Originally adapted from https://github.com/keras-team/keras/issues/1401
# https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
# Requires Keras and Tensorflow backend


#from keras.layers import containers
from keras.models import Sequential, Model,load_model
from keras.layers import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model,to_categorical,print_summary
from keras.backend import argmax
from tensorflow import InteractiveSession
import numpy as np
import os.path

def preprocess(text,token):
	text = token.texts_to_sequences(text)
	text = np.array(text)
	text = pad_sequences(text)
	text = to_categorical(text,len(token.word_index.items())+1)
	(N,sequence,voc) = text.shape
	return text, N, sequence, voc

def load_dataset(filename,k = 0,token=None):
	corpus = open(filename).read().lower().splitlines()
	if token is None:
		token = Tokenizer()
		token.fit_on_texts(corpus)
	corpus, N, sequence, voc = preprocess(corpus,token)
	## Extracting Training data and initializing some variables for the model
	x = corpus[0:2*k:2] # extract every second item from the list
	t = corpus[1:2*k:2]
	return x,t,N,sequence,voc,token

def build_autoencoder(l1,l2,voc,x,e):
	autoencoder = Sequential()
	autoencoder.add(LSTM(l1,input_shape = (None,voc), return_sequences=True))
	#autoencoder.add(LSTM(l2,return_sequences=True))
	autoencoder.add(LSTM( voc, return_sequences=True))
	autoencoder.add(Dense(voc, activation='softmax'))
	autoencoder.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics = ['acc'])
	early_stopping = EarlyStopping(monitor='val_loss', patience=4)
	autoencoder.fit(x,x, epochs = e, validation_split=0.2,callbacks=[early_stopping])
	return autoencoder

def build_classifier(source_model,voc,x,t,e,l1,l2):
	classifier = Sequential()
	#Think we may need to work with a masking layer here to avoid the zeros
	classifier.add(LSTM(l1,input_shape=(None,voc)))
	#classifier.add(LSTM(l2,return_sequences=True))
	classifier.layers[0].set_weights(source_model.layers[0].get_weights())
	#classifier.layers[1].set_weights(source_model.layers[1].get_weights())
	classifier.add(Dense(voc,activation='softmax'))
	classifier.compile(loss='categorical_crossentropy',optimizer='Adam',metrics = ['acc'])
	classifier.fit(x,t, epochs = e)
	# We should look at fine tuning as well, basically evaluate  on train_t
	return classifier

def sequences_to_text(token,x):
	reverse_word_dict = dict(map(reversed,token.word_index.items()))
	InteractiveSession()
	from_categorical = lambda y: argmax(y,axis=-1).eval()
	seqs_to_words = lambda y: list(map(reverse_word_dict.get,from_categorical(y))) 
	words_to_sentence = lambda y: ' '.join(filter(None,y))
	word_matrix = list(map(seqs_to_words,x))
	sentence_list = list(map(words_to_sentence,word_matrix))
	return sentence_list

#Dimensionality reduction in encoder1 and encoder 2
latent_dimension1 = 140 
latent_dimension2 = 50
epochs = 2
load_data = True
file_id = 'Autoencoder_' +str(epochs)+'_'+ str(latent_dimension1) + '_' + str(latent_dimension2) +'.h5'


if load_data:
	# This function fetches the dataset from the file and fills both X and T with k number of datapoints
	train_x, train_t,train_N,train_sequence,voc,train_token = load_dataset("qa.894.raw.train.txt",100)
	test_x, test_t,test_N,test_sequence,_,_ = load_dataset("qa.894.raw.test.txt",100,train_token)

## Build and train Autoencoder
if os.path.exists(file_id):
	print('\n Model with these parameters found, loading model \n')
	autoencoder = load_model(file_id)
else:
	print('\n No model with these parameters was found, building new model.\n')
	autoencoder = build_autoencoder(latent_dimension1,latent_dimension2,voc,train_x,epochs)
	#autoencoder.save(file_id)
print('Autoencoder parameters')
#autoencoder.summary()
#print_summary(autoencoder)


classifier = build_classifier(autoencoder,voc,train_x,train_t,epochs,latent_dimension1,latent_dimension2)
print(classifier.evaluate(test_x,test_t))
answer = classifier.predict(test_x)
print(answer[0])
print(sequences_to_text(token,answer))
