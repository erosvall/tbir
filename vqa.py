# Originally adapted from https://github.com/keras-team/keras/issues/1401
# https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
# Requires Keras and Tensorflow backend

from keras.models import load_model, Model
from keras.layers import Dense, Embedding, Input, Dropout, concatenate, RepeatVector
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
import preprocesser as prep
import postprocesser as postp
import model_full as full
import model_text as text
import os.path
import argparse
import sys

def model(epochs,ld1,batch,load,textonly):
    filestart = 'Text' if textonly else 'Full'
    file_id = filestart + '_Question_Answerer_' + str(epochs) + '_' + str(ld1) + '.h5'

    # This function fetches the dataset from the file and fills both X and T with k number of datapoints
    train_x,train_x_cat, train_imgs, train_t, train_t_cat,train_N, train_sequence, voc, train_token = prep.load_dataset("qa.894.raw.train.txt", 6795,img_filename="img_features.csv")
    test_x,test_x_cat, test_imgs, test_t,test_t_cat, test_N, test_sequence, _, _ = prep.load_dataset("qa.894.raw.test.txt", 6795 , train_token, img_filename="img_features.csv")

    # Build and train Question Answerer
    if load:
        model = load_model(load)
        print('\nLoading Question Answerer model from file: ' + load + ' \n')
        if args.improve:
            if textonly:
                model = text.train_model(model,train_x,train_x_cat,train_t_cat,epochs,batch,ld1)
            else:
                model = full.train_model(model,train_x,train_imgs,train_x_cat,train_t_cat,epochs,batch,ld1)
    elif os.path.exists(file_id):
        print('\nQuestion Answerer model with these parameters found, loading model from file: ' + file_id + '\n')
        model = load_model(file_id)
    else:
        print('\nNo question answerer model with these parameters was found, building new model.\n')
        if textonly:
            model = text.build_model(ld1,voc)
            model = text.train_model(model,train_x,train_x_cat,train_t_cat,epochs,batch,ld1)
        else:
            model = full.build_model(ld1,voc,train_imgs.shape[1])
            model = full.train_model(model,train_x,train_imgs,train_x_cat,train_t_cat,epochs,batch,ld1)
        model.save(file_id)
        print('\nModel saved to: ' + file_id)

    print('\nEvaluating question answerer on test data')
    if textonly:
        qa_result = model.evaluate(test_x, [test_t_cat, test_x_cat], batch_size=batch)
        [qa_answer,qa_question] = model.predict(test_x, batch_size=batch)
    else:
        qa_result = model.evaluate([test_x,test_imgs], [test_t_cat,test_x_cat], batch_size=batch)
        [qa_answer,qa_question] = model.predict([test_x,test_imgs], batch_size=batch)
        
    print('Loss: ' + str(qa_result[0]))
    print('Accuracy: ' + str(qa_result[1]))

    return test_x, test_t, qa_answer, qa_question, train_token

def main(argv=None):
    # EXAMPLES
    argparser = argparse.ArgumentParser(description='A visual question answerer.')
    # optional arguments
    argparser.add_argument('--load', type=str,
                           help='Filename of existing model')
    argparser.add_argument('--e', type=int,
                           help='Number of epochs, default 1')
    argparser.add_argument('--ld1', type=int,   
                           help='Latent dimension 1, default 512')
    argparser.add_argument('--b', type=int,
                           help='Batch size, default 32')
    argparser.add_argument('--wups', action="store_true",
                           help='Compute the WUPS Score')
    argparser.add_argument('--textonly', action="store_true",
                           help='Ignore the images')
    argparser.add_argument('--improve', action="store_true",
                           help='Further train the loaded model')
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

    test_x, test_t, qa_answer, qa_question, train_token = model(epochs,ld1,batch,args.load,args.textonly)

    postp.print_compare(test_x,test_t,qa_answer,qa_question,10,train_token,args.wups)

if __name__ == "__main__":
    sys.exit(main())
