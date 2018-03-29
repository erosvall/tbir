# Originally adapted from https://github.com/keras-team/keras/issues/1401
# https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
# Requires Keras and Tensorflow backend


#from keras.layers import containers
from keras.models import Sequential, Model,load_model
from keras.layers import Dense, Dropout,Masking,Embedding,Flatten
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model,to_categorical,print_summary
from keras.backend import argmax
from keras.callbacks import ModelCheckpoint
from tensorflow import InteractiveSession
from keras import regularizers
from gensim.models import Word2Vec
import numpy as np
import os.path

def preprocess(text,token):
    x = token.texts_to_sequences(text)
    x = pad_sequences(x)
    #x = to_categorical(x,len(token.word_index.items())+1)
    voc = len(token.word_index.items())+1
    y = token.texts_to_matrix(text,mode='binary')
    words = []
    for line in text:
        words.append(line.split())
    model = Word2Vec(words,size=300)
    print(model)
    embedding_matrix = np.zeros((voc, 300))
    correct = 0
    false = 0
    for word, i in token.word_index.items():
        if word in model:
            correct += 1
            embedding_matrix[i] = model[word]
        else:
            false += 1
            embedding_matrix[i] = np.random.rand(1, 300)[0]
    print('CORRECT')
    print(correct)
    print('FALSE')
    print(false)
    print('SHAPE' + str(x.shape)+str(voc))
    (N,sequence) = x.shape
    print('TEXT')
    print(x)
    return x, y, embedding_matrix, N, sequence, voc

def load_dataset(filename,k = 0,token=None):
    corpus = open(filename).read().lower().splitlines()
    if token is None:
        token = Tokenizer(oov_token = '~')
        token.fit_on_texts(corpus)
    corpus,y,embedding_matrix, N, sequence, voc = preprocess(corpus,token)
    ## Extracting Training data and initializing some variables for the model
    x = corpus[0:2*k:2]
    y = y[0:2*k:2]
    # extract every second item from the list
    t = corpus[1:2*k:2]
    return x,y,t,N,sequence,voc,token,embedding_matrix

def build_autoencoder(l1,l2,voc,seq,x,y,e,batch,embedding_matrix):
    callback = ModelCheckpoint('Autoencoder_{epoch:02d}-{loss:.3f}_'+ str(l1) + '_' + str(l2) +'.h5',monitor='loss',save_best_only=True)
    autoencoder = Sequential()
    autoencoder.add(Embedding(input_dim=voc,output_dim=300,input_length=seq,input_shape = (seq,),mask_zero=True,weights=[embedding_matrix]))
    autoencoder.add(LSTM(l1, return_sequences=True))
    autoencoder.add(LSTM(l2,return_sequences=True))
    autoencoder.add(LSTM( voc, return_sequences=True))
    autoencoder.add(Flatten())
    autoencoder.add(Dense(voc, activation='softmax'))
    autoencoder.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['acc'])
    autoencoder.fit(x,y, epochs = e,callbacks=[callback],batch_size = batch)
    return autoencoder

def build_classifier(source_model,voc,x,t,e,l1,l2,batch):
    classifier = Sequential()
    #Think we may need to work with a masking layer here to avoid the zeros
    classifier.add(LSTM(l1,return_sequences=True, input_shape = (None,voc)))
    classifier.add(LSTM(l2,return_sequences=False))
    classifier.layers[0].set_weights(source_model.layers[0].get_weights())
    classifier.layers[0].trainable = False # Ensure that we don't change representation weights
    classifier.layers[1].set_weights(source_model.layers[1].get_weights())
    classifier.layers[1].trainable = False # Ensure that we don't change representation weights
    classifier.add(Dense(voc,activation='softmax', kernel_regularizer=regularizers.l1(0.))) 
    classifier.compile(loss='categorical_crossentropy',optimizer='adam',metrics = ['categorical_accuracy'])
    classifier.fit(x,t, epochs = e,batch_size = batch)
    # We should look at fine tuning as well, basically evaluate  on train_t
    return classifier


def sequences_to_text(token,x):
    print('Converting to text...')
    reverse_word_dict = dict(map(reversed,token.word_index.items()))
    InteractiveSession()
    from_categorical = lambda y: argmax(y,axis=-1).eval()
    seqs_to_words = lambda y: list(map(reverse_word_dict.get,from_categorical(y))) 
    words_to_sentence = lambda y: ' '.join(filter(None,y))
    word_matrix = list(map(seqs_to_words,x))
    sentence_list = list(map(words_to_sentence,word_matrix))
    return '\n'.join(sentence_list)


def to_text(token,x):
    print('Converting to text...')
    reverse_word_dict = dict(map(reversed,token.word_index.items()))
    InteractiveSession()
    # from_categorical = lambda y: argmax(y,axis=-1).eval()
    seqs_to_words = lambda y: list(map(reverse_word_dict.get,y)) 
    words_to_sentence = lambda y: ' '.join(filter(None,y))
    word_matrix = list(map(seqs_to_words,x))
    sentence_list = list(map(words_to_sentence,word_matrix))
    return '\n'.join(sentence_list)
    
def matrix_to_text(token,x):
    print('Converting to text vector...')
    reverse_word_dict = dict(map(reversed,token.word_index.items()))
    InteractiveSession()
    seqs_to_words = lambda y: list(map(reverse_word_dict.get,argmax(y,axis=-1).eval()))
    return seqs_to_words(x)
#Dimensionality reduction in encoder1 and encoder 2

batch = 512
ld1 = 140 
ld2 = 50
epochs = 24
file_id = 'Autoencoder_' +str(epochs)+'_'+ str(ld1) + '_' + str(ld2) +'.h5'



# This function fetches the dataset from the file and fills both X and T with k number of datapoints
train_x, train_y,train_t,train_N,train_sequence,voc,train_token,embedding_matrix = load_dataset("qa.894.raw.train.txt",6795)
# test_x, test_y,test_t,test_N,test_sequence,_,_,_ = load_dataset("qa.894.raw.test.txt",6795,train_token)

if True:
    ## Build and train Autoencoder
    if os.path.exists(file_id):
        print('\n Model with these parameters found, loading model \n')
        autoencoder = load_model(file_id)
    else:
        print('\n No model with these parameters was found, building new model.\n')
        autoencoder = build_autoencoder(ld1,ld2,voc,train_sequence,train_x,train_y,epochs,batch,embedding_matrix)
        autoencoder.save(file_id)
    print('Autoencoder parameters')
    autoencoder.summary()

    # new_test_t = test_t[:,-1,:]

    # classifier = build_classifier(autoencoder,voc,train_x,train_t[:,-1,:],50,ld1,ld2,batch)
    print(autoencoder.evaluate(train_x,train_x,batch_size=batch))
    answer = autoencoder.predict(train_x,batch_size=batch)
    rand = np.random.choice(4000,10)
    print('ORIGINAL')
    print(test_x)
    print(to_text(train_token,test_x[rand]))
    print('PREDICTION')
    print(answer)
    print(to_text(train_token,answer    [rand]))