# Originally adapted from https://github.com/keras-team/keras/issues/1401
# https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
# Requires Keras and Tensorflow backend

from keras.models import load_model, Model
from keras.layers import Dense, Embedding, Input, Dropout, concatenate, RepeatVector
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
from tensorflow import InteractiveSession
import preprocesser as prep
import postprocesser as postp
import model_full as full
import model_text as text
import model_visual as visual
import os.path
import argparse
import sys

def model(epochs,drop,ae_weight,ld1,batch,load,test,textonly,visualonly,improve,checkpoint):
    filestart = 'Text' if textonly else 'Visual' if visualonly else 'Full'
    file_id = filestart + '_Question_Answerer_' + str(epochs) + '_' + str(ld1) + '_' + str(ae_weight) + '.h5'

    # This function fetches the dataset from the file and fills both X and T with k number of datapoints
    train_x,train_x_cat, train_imgs, train_t, train_t_cat,train_N, train_sequence, voc, train_token = prep.load_dataset("qa.894.raw.train.txt", 6795,img_filename="img_features.csv")
    test_x,test_x_cat, test_imgs, test_t,test_t_cat, test_N, test_sequence, _, _ = prep.load_dataset(test, 6795 , train_token, img_filename="img_features.csv")

    # Build and train Question Answerer
    if load:
        model = load_model(load)
        print('\nLoading Question Answerer model from file: ' + load + ' \n')
        if improve:
            if textonly:
                model = text.train_model(model,train_x,train_x_cat,train_t_cat,epochs,batch,ld1,ae_weight,checkpoint)
            elif visualonly:
                model = visual.train_model(model,train_x,train_imgs,train_t_cat,epochs,batch,ld1,ae_weight,checkpoint)
            else:
                model = full.train_model(model,train_x,train_imgs,train_x_cat,train_t_cat,epochs,batch,ld1,ae_weight,checkpoint)
    elif os.path.exists(file_id):
        print('\nQuestion Answerer model with these parameters found, loading model from file: ' + file_id + '\n')
        model = load_model(file_id)
    else:
        print('\nNo question answerer model with these parameters was found, building new model.\n')
        if textonly:
            model = text.build_model(drop,ae_weight,ld1,voc)
            model = text.train_model(model,train_x,train_x_cat,train_t_cat,epochs,batch,ld1,ae_weight,checkpoint)
        elif visualonly:
            model = visual.build_model(drop,ld1,voc,train_imgs.shape[1])
            model = visual.train_model(model,train_x,train_imgs,train_t_cat,epochs,batch,ld1,ae_weight,checkpoint)
        else:
            model = full.build_model(drop,ae_weight,ld1,voc,train_imgs.shape[1])
            model = full.train_model(model,train_x,train_imgs,train_x_cat,train_t_cat,epochs,batch,ld1,ae_weight,checkpoint)
        model.save(file_id)
        print('\nModel saved to: ' + file_id)

    print('\nEvaluating question answerer on test data')
    if textonly:
        qa_result = model.evaluate(test_x, [test_t_cat, test_x_cat], batch_size=batch)
        [qa_answer,qa_question] = model.predict(test_x, batch_size=batch)
    elif visualonly:
        qa_result = model.evaluate([test_x,test_imgs],test_t_cat, batch_size=batch)
        qa_answer = model.predict([test_x,test_imgs], batch_size=batch)
        qa_question = None
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
                           help='Filename of existing model, default None')
    argparser.add_argument('--test', type=str,
                           help='Filename of test data, default qa.894.raw.test.txt')
    argparser.add_argument('--e', type=int,
                           help='Number of epochs, default 1')
    argparser.add_argument('--ld1', type=int,   
                           help='Latent dimension 1, default 512')
    argparser.add_argument('--b', type=int,
                           help='Batch size, default 32')
    argparser.add_argument('--drop', type=float,
                           help='Dropout percentage, default 0.5')
    argparser.add_argument('--aeweight', type=float,
                           help='Weight of the autoencoder loss function compared to the answer loss function, default 1.0')
    argparser.add_argument('--wups', action="store_true",
                           help='Compute the WUPS Score')
    argparser.add_argument('--textonly', action="store_true",
                           help='Ignore the images')
    argparser.add_argument('--visualonly', action="store_true",
                           help='Without autoencoder')
    argparser.add_argument('--improve', action="store_true",
                           help='Further train the loaded model')
    argparser.add_argument('--checkpoint', action="store_true",
                           help='Save at every epoch')

    args = argparser.parse_args(argv)


    # Hyper Parameters
    ld1 = 512
    epochs = 1
    batch = 32
    nbtest = 50
    drop = 0.5
    ae_weight = 1.0
    test = "qa.894.raw.test.txt"

    if args.ld1:
        ld1 = args.ld1
    if args.e:
        epochs = args.e
    if args.b:
        batch = args.b
    if args.drop:
        drop = args.drop
    if args.aeweight:
        ae_weight = args.aeweight
    if args.test:
        test = args.test

    print('--e Number of epochs: ' + str(epochs))
    print('--ld1 Latent dimension 1: ' + str(ld1))
    print('--b Batch size: ' + str(batch))
    print('--drop Dropout percentage: ' + str(drop))
    print('--aeweight Autoencoder Loss Weight: ' + str(ae_weight))
    print('')

    InteractiveSession()

    test_x, test_t, qa_answer, qa_question, train_token = model(epochs,drop,ae_weight,ld1,batch,args.load,test,args.textonly,args.visualonly,args.improve,args.checkpoint)

    if args.wups:
        postp.print_wups_acc(test_t,qa_answer,train_token)
        if not args.visualonly:
            postp.print_ae_acc(test_x,qa_question,train_token)
    postp.print_compare(test_x,test_t,qa_answer,qa_question,nbtest,train_token)

if __name__ == "__main__":
    sys.exit(main())
