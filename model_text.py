from keras.models import load_model, Model
from keras.layers import Dense, Embedding, Input, Dropout, concatenate, RepeatVector
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
import preprocesser as prep
import postprocesser as postp
import os.path
import argparse
import sys

def build_model(l1,voc):
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
    model = Model(
        inputs = word_input, 
        outputs = [output, autoencoder_output])
    model.compile(
        loss = 'categorical_crossentropy', 
        optimizer = 'adam', 
        metrics = ['categorical_accuracy'],
        loss_weights=[1., 1.])

    # plot_model(model, to_file='text_classifier_w_autoencoder.png')
    return model

def train_model(model,x,x_cat,t_cat,e,batch,l1):
    print('Training...\n')
    checkpoint = ModelCheckpoint('QA_e{epoch:02d}-l{loss:.3f}-cl{out_loss:.3f}-ael{AE_out_loss:.3f}-cac{out_categorical_accuracy:.3f}-aeac{AE_out_categorical_accuracy:.3f}-vl{val_loss:.3f}-vcl{val_out_loss:.3f}-vael{val_AE_out_loss:.3f}-vcac{val_out_categorical_accuracy:.3f}-vaeac{val_AE_out_categorical_accuracy:.3f}_' + str(l1) + '.h5', monitor='val_loss',save_best_only=False)
    model.fit(
        x,
        [t_cat, x_cat], 
        epochs = e, 
        batch_size = batch,
        callbacks=[checkpoint],
        validation_split = 0.1)
    return model