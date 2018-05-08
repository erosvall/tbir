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
import numpy as np
import os.path
import argparse
import sys
import preprocesser as prep
import postprocesser as postp

def build_text_classifier(x,x_cat, t_cat, e,l1, voc, batch):
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
    encoder, h_state, c_state = LSTM(
        l1,
        name='encoder',
        return_state = True
        )(word_embedding)
    repeat_vector = RepeatVector(30)(encoder)
    decoder= LSTM(
        voc,
        return_sequences = True,
        name='decoder',
        go_backwards = True
        )(repeat_vector)# , initial_state = [ h_state, c_state ]
    autoencoder_output = Dense(
        voc,
        activation = 'softmax',
        name = 'AE_out'
        )(decoder)


    # We merge the model, add a dropout to combat some overfitting and fit.
    merged = concatenate([word_encoding, encoder]) # Concatenate an Autoencoder hidden layer here
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
        name = 'out')(answer_layer)
    classifier = Model(
        inputs = word_input, 
        outputs = [output, autoencoder_output])
    classifier.compile(
        loss = 'categorical_crossentropy', 
        optimizer = 'adam', 
        metrics = ['categorical_accuracy'],
        loss_weights=[1., 1.])

    # plot_model(classifier, to_file='text_classifier_w_autoencoder.png')

    print('Training...\n')
    classifier.fit(
        x,
        [t_cat, x_cat], 
        epochs = e, 
        batch_size = batch,
        validation_split = 0.1)
    return classifier

def main(argv=None):
    # EXAMPLES
    argparser = argparse.ArgumentParser(description='An inference engine for problog programs and bayesian networks.')
    # optional arguments
    argparser.add_argument('--qa', type=str,
                           help='Filename of existing classifier model')
    argparser.add_argument('--e', type=int,
                           help='Number of epochs, default 1')
    argparser.add_argument('--ld1', type=int,   
                           help='Latent dimension 1, default 512')
    argparser.add_argument('--b', type=int,
                           help='Batch size, default 256')
    argparser.add_argument('--wups', action="store_true",
                           help='Compute the WUPS Score')
    args = argparser.parse_args(argv)


    # Hyper Parameters
    ld1 = 512
    epochs = 1
    batch = 256

    if args.ld1:
        ld1 = args.ld1
    if args.e:
        epochs = args.e
    if args.b:
        batch = args.b

    print('--e Number of epochs: ' + str(epochs))
    print('--ld1 Latent dimension 1: ' + str(ld1))
    print('--b Batch size: ' + str(batch))

    qa_file_id = 'Text_Question_Answerer_' + str(epochs) + '_' + str(ld1) + '.h5'

    # This function fetches the dataset from the file and fills both X and T with k number of datapoints
    train_x,train_x_cat, train_imgs, train_t, train_t_cat,train_N, train_sequence, voc, train_token = prep.load_dataset("qa.894.raw.train.txt", 6795,img_filename="img_features.csv")
    test_x,test_x_cat, test_imgs, test_t,test_t_cat, test_N, test_sequence, _, _ = prep.load_dataset("qa.894.raw.test.txt", 6795 , train_token, img_filename="img_features.csv")


    # Build and train Question Answerer
    if args.qa:
        print('\nLoading Question Answerer model from file: ' + args.qa + ' \n')
        classifier = load_model(args.qa)
    elif os.path.exists(qa_file_id):
        print('\nQuestion Answerer model with these parameters found, loading model from file: ' + qa_file_id + '\n')
        classifier = load_model(qa_file_id)
    else:
        print('\nNo question answerer model with these parameters was found, building new model.\n')
        classifier = build_text_classifier(train_x,train_x_cat, train_t_cat, epochs, ld1, voc, batch)
        classifier.save(qa_file_id)
        print('\nModel saved to: ' + qa_file_id)

    

    print('\nEvaluating question answerer on test data')
    qa_result = classifier.evaluate(test_x, [test_t_cat, test_x_cat], batch_size = batch)
    [qa_answer,qa_question] = classifier.predict(test_x, batch_size = batch)
    print(qa_answer.shape)
    print(qa_question.shape)
    print('Loss: ' + str(qa_result[0]))
    print('Accuracy: ' + str(qa_result[1]))

    postp.print_compare(test_x,test_t_cat,qa_answer,20,train_token,args.wups)
    postp.print_auto(test_x,qa_question,20,train_token)

if __name__ == "__main__":
    sys.exit(main())
