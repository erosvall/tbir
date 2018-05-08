# Originally adapted from https://github.com/keras-team/keras/issues/1401
# https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
# Requires Keras and Tensorflow backend

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
import preprocesser as prep
import postprocesser as postp
import numpy as np
import os.path
import argparse
import sys
import time

def build_classifier(words, images, t, e,l1, voc, batch):
    print('Building model...\n')
    # Build text based input. embedding and LSTM layer
    # The Input layer is only there for convenience and doesn't do anything
    # The Embedding layer takes a 2D input (batch_size, sequence_length) and
    # outputs a 3D tensor (batch_size,sequence_length,embedding_dim) for the 
    # LSTM. The LSTM and Embedding layer match dimensions for convience. The
    # LSTM is configured to only pass information forward at the end of the 
    # sequence. 
    word_input = Input(shape=(30,),name='Text_input')
    word_embedding = Embedding(
        input_dim = voc,
        output_dim = l1,
        mask_zero = True
        )(word_input)
    word_encoding = LSTM(
        l1,
        name = 'Text_features'
        )(word_embedding) 
    # Autoencoder part. It uses its own auxiliary output for optimization.
    # It borrows the same textual input as the other LSTM layer, in addition
    # does it concatenate with the rest of the model.
    encoder= LSTM(
        l1,
        name='encoder'
        )(word_embedding)
    repeat_vector = RepeatVector(30)(encoder)
    decoder= LSTM(
        voc,
        name='decoder',
        go_backwards = True
        )(repeat_vector)
    autoencoder_output = Dense(
        30,
        activation = 'softmax',
        name = 'Autoencoder_output'
        )(decoder)


    # Construtct the Image input part. Since no feature extraction 
    # takes place we basically just run ahead here
    visual_input = Input(shape=(images.shape[1],),name='Image_input')
    visual_encoding = Dense(images.shape[1],name='Image_features')(visual_input) 



    # We merge the model, add a dropout to combat some overfitting and fit.
    merged = concatenate([word_encoding,visual_encoding, encoder]) # Concatenate an Autoencoder hidden layer here
    # We might want a LSTM layer with return sequence set to true here?
    dropout = Dropout(0.5)(merged)
    repeat_vector = RepeatVector(11)(dropout)
    answer_layer = LSTM(
    voc,
    return_sequences = True,
    name = 'answer_sequence'
    )(repeat_vector) # Ability to answer multiple answers

    output = Dense(
        voc,
        activation = 'softmax',
        name = 'Output_layer')(answer_layer)
    classifier = Model(
        inputs = [word_input, visual_input], 
        outputs = [output, autoencoder_output])
    classifier.compile(
        loss = 'categorical_crossentropy', 
        optimizer = 'adam', 
        metrics = ['categorical_accuracy'],
        loss_weights=[1., 1.])

    # plot_model(classifier, to_file='classifier_w_autoencoder.png')

    print('Training...\n')
    classifier.fit(
        [words, images],
        [t, np.flip(words,axis=1)], 
        epochs = e, 
        batch_size = batch,
        validation_split = 0.1)
    return classifier

def build_classifier1(words, images, t, e,l1, voc, batch):
    print('Building model...\n')
    # Build text based input. embedding and LSTM layer
    # The Input layer is only there for convenience and doesn't do anything
    # The Embedding layer takes a 2D input (batch_size, sequence_length) and
    # outputs a 3D tensor (batch_size,sequence_length,embedding_dim) for the 
    # LSTM. The LSTM and Embedding layer match dimensions for convience. The
    # LSTM is configured to only pass information forward at the end of the 
    # sequence. 
    word_input = Input(shape=(30,),name='Text_input')
    word_embedding = Embedding(
        input_dim = voc,
        output_dim = l1,
        mask_zero = True
        )(word_input)
    word_encoding = LSTM(
        l1,
        name = 'Text_features'
        )(word_embedding) 
    
    # Construtct the Image input part. Since no feature extraction 
    # takes place we basically just run ahead here
    visual_input = Input(shape=(images.shape[1],),name='Image_input')
    visual_encoding = Dense(images.shape[1],name='Image_features')(visual_input) 



    # We merge the model, add a dropout to combat some overfitting and fit.
    merged = concatenate([word_encoding,visual_encoding]) # Concatenate an Autoencoder hidden layer here
    # We might want a LSTM layer with return sequence set to true here?
    dropout = Dropout(0.5)(merged)
    repeat_vector = RepeatVector(11)(dropout)
    answer_layer = LSTM(
    voc,
    return_sequences = True,
    name = 'answer_sequence'
    )(repeat_vector) # Ability to answer multiple answers

    output = Dense(
        voc,
        activation = 'softmax',
        name = 'Output_layer')(answer_layer)
    classifier = Model(
        inputs = [word_input, visual_input], 
        outputs = [output])
    classifier.compile(
        loss = 'categorical_crossentropy', 
        optimizer = 'adam', 
        metrics = ['categorical_accuracy'],
        )

    # plot_model(classifier, to_file='classifier.png')

    print('Training...\n')
    classifier.fit(
        [words, images],
        t, 
        epochs = e, 
        batch_size = batch,
        validation_split = 0.1)
    return classifier

def main(argv=None):
    # EXAMPLES
    argparser = argparse.ArgumentParser(description='A visual question answerer.')
    # optional arguments
    argparser.add_argument('--qa', type=str,
                           help='Filename of existing classifier model')
    argparser.add_argument('--e', type=int,
                           help='Number of epochs, default 1')
    argparser.add_argument('--ld1', type=int,   
                           help='Latent dimension 1, default 512')
    argparser.add_argument('--b', type=int,
                           help='Batch size, default 32')
    argparser.add_argument('--wups', action="store_true",
                           help='Compute the WUPS Score')
    args = argparser.parse_args(argv)


    # Hyper Parameters
    ld1 = 512
    epochs = 1
    batch = 32

    if args.ld1:
        ld1 = args.ld1
    if args.e:
        epochs = args.e
    if args.b:
        batch = args.b

    print('--e Number of epochs: ' + str(epochs))
    print('--ld1 Latent dimension 1: ' + str(ld1))
    print('--b Batch size: ' + str(batch))

    qa_file_id = 'Enc_Question_Answerer_' + str(epochs) + '_' + str(ld1) + '.h5'

    # This function fetches the dataset from the file and fills both X and T with k number of datapoints
    train_x, train_imgs, train_t, train_N, train_sequence, voc, train_token = prep.load_dataset("qa.894.raw.train.txt", 6795,img_filename="img_features.csv")
    test_x, test_imgs, test_t, test_N, test_sequence, _, _ = prep.load_dataset("qa.894.raw.test.txt", 6795 , train_token, img_filename="img_features.csv")


    # Build and train Question Answerer
    if args.qa:
        print('\nLoading Question Answerer model from file: ' + args.qa + ' \n')
        classifier = load_model(args.qa)
    elif os.path.exists(qa_file_id):
        print('\nQuestion Answerer model with these parameters found, loading model from file: ' + qa_file_id + '\n')
        classifier = load_model(qa_file_id)
    else:
        print('\nNo question answerer model with these parameters was found, building new model.\n')
        classifier = build_classifier1(train_x, train_imgs, train_t, epochs, ld1, voc, batch)
        classifier.save(qa_file_id)
        print('\nModel saved to: ' + qa_file_id)

    

    print('\nEvaluating question answerer on test data')
    qa_result = classifier.evaluate([test_x,test_imgs], test_t, batch_size = batch)
    qa_answer = classifier.predict([test_x,test_imgs], batch_size = batch)
    print('Loss: ' + str(qa_result[0]))
    print('Accuracy: ' + str(qa_result[1]))

    postp.print_compare(test_x,test_t,qa_answer,100,train_token,args.wups)


if __name__ == "__main__":
    sys.exit(main())
