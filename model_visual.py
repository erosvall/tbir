from keras.models import load_model, Model
from keras.layers import Dense, Embedding, Input, Dropout, concatenate, RepeatVector
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
import preprocesser as prep
import postprocesser as postp
import os.path
import argparse
import sys

def build_model(l1, voc, img_dim):
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
    visual_input = Input(shape=(img_dim,),name='Image_input')
    visual_encoding = Dense(img_dim,name='Image_features')(visual_input) 



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
        name = 'out')(answer_layer)
    model = Model(
        inputs = [word_input, visual_input], 
        outputs = [output])
    model.compile(
        loss = 'categorical_crossentropy', 
        optimizer = 'adam', 
        metrics = ['categorical_accuracy'],
        )

    # plot_model(model, to_file='model.png')
    return model

def train_model(model,x,images,t_cat,e,batch,l1):
    print('Training...\n')
    checkpoint = ModelCheckpoint('VQA_e{epoch:02d}-l{loss:.3f}-acc{categorical_accuracy:.3f}-vl{val_loss:.3f}-vacc{val_categorical_accuracy:.3f}_' + str(l1) + '.h5', monitor='val_loss',save_best_only=False)
    model.fit(
        [x, images],
        t_cat, 
        epochs = e, 
        batch_size = batch,
        callbacks=[checkpoint],
        validation_split = 0.1)
    return model