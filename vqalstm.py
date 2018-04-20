# Originally adapted from https://github.com/keras-team/keras/issues/1401
# https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
# Requires Keras and Tensorflow backend

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.backend import argmax
from keras.callbacks import ModelCheckpoint
from tensorflow import InteractiveSession
from keras import regularizers
import os.path
import argparse
import sys

def load_cnn(filename):
    # Returns a matrix where each row corresponds to imageXXXX, note that index determines image number
    images = open(filename).read().splitlines()
    for i,img in enumerate(images):
        images[i] = img.split(',')
        images[i][0] = int(images[i][0][5:])
    images = np.asarray(images).astype(float)
    images = images[images[:,0].argsort()]
    return images[:,1:]

    

def preprocess(text, token):
    text = token.texts_to_sequences(text)
    text = pad_sequences(text)
    text = to_categorical(text, len(token.word_index.items())+1)
    (N, sequence, voc) = text.shape
    return text, N, sequence, voc


def load_dataset(filename, k=0, token=None):
    corpus = open(filename).read().lower().splitlines()
    if token is None:
        token = Tokenizer()#oov_token='~')
        token.fit_on_texts(corpus)
    corpus, N, sequence, voc = preprocess(corpus, token)
    # Extracting Training data and initializing some variables for the model
    x = corpus[0:2*k:2]  # extract every second item from the list
    t = corpus[1:2*k:2]
    return x, t, N, sequence, voc, token


def build_classifier(voc, x, t, e, l1, batch):
    classifier = Sequential()
    classifier.add(Embedding(input_shape=(None,voc),output_dim=(l1),mask_zero = True))
    classifier.add(LSTM(l1))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(voc, activation='softmax'))
    classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    classifier.fit(x, t, epochs=e, batch_size=batch)
    return classifier


def sequences_to_text(token, x):
    print('Converting to text...')
    reverse_word_dict = dict(map(reversed, token.word_index.items()))
    InteractiveSession()
    from_categorical = lambda y: argmax(y, axis=-1).eval()
    seqs_to_words = lambda y: list(map(reverse_word_dict.get, from_categorical(y)))
    words_to_sentence = lambda y: ' '.join(filter(None, y))
    word_matrix = list(map(seqs_to_words, x))
    sentence_list = list(map(words_to_sentence, word_matrix))
    return '\n'.join(sentence_list)


def matrix_to_text(token, x):
    print('Converting to text vector...')
    reverse_word_dict = dict(map(reversed, token.word_index.items()))
    InteractiveSession()
    seqs_to_words = lambda y: list(map(reverse_word_dict.get, argmax(y,axis=-1).eval()))
    return seqs_to_words(x)


def main(argv=None):
    # EXAMPLES
    argparser = argparse.ArgumentParser(description='An inference engine for problog programs and bayesian networks.')
    # optional arguments
    argparser.add_argument('--qa', type=str,
                           help='Filename of existing classifier model')
    argparser.add_argument('--e', type=int,
                           help='Number of epochs, default 1')
    argparser.add_argument('--ld1', type=int,
                           help='Latent dimension 1, default 140')
    argparser.add_argument('--b', type=int,
                           help='Batch size, default 32')
    args = argparser.parse_args(argv)


    # Dimensionality reduction in encoder1 
    ld1 = 140
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

    qa_file_id = 'Question_Answerer_' + str(epochs) + '_' + str(ld1) + '_' + str(ld2) + '.h5'

    # This function fetches the dataset from the file and fills both X and T with k number of datapoints
    train_x, train_t, train_N, train_sequence, voc, train_token = load_dataset("qa.894.raw.train.txt", 6795)
    test_x, test_t, test_N, test_sequence, _, _ = load_dataset("qa.894.raw.test.txt", 6795, train_token)


    # Build and train Question Answerer
    if args.qa:
        print('\nLoading Question Answerer model from file: ' + args.qa + ' \n')
        classifier = load_model(args.qa)
    elif os.path.exists(qa_file_id):
        print('\nQuestion Answerer model with these parameters found, loading model from file: ' + qa_file_id + '\n')
        classifier = load_model(qa_file_id)
    else:
        print('\nNo question answerer model with these parameters was found, building new model.\n')
        classifier = build_classifier(autoencoder, voc, train_x, train_t, epochs, ld1, ld2, batch)
        classifier.save(qa_file_id)
        print('\nModel saved to: ' + qa_file_id)

    print('\nQuestion answerer parameters')
    classifier.summary()

    rand = range(10)  # np.random.choice(4000, 10)

    print('\nEvaluating question answerer on test data')
    qa_result = classifier.evaluate(test_x, test_t, batch_size=batch)
    qa_answer = classifier.predict(test_x, batch_size=batch)
    print('Loss: ' + str(qa_result[0]))
    print('Accuracy: ' + str(qa_result[1]))

    print('\nFirst 10 questions:')
    print(sequences_to_text(train_token, test_x[rand]))
    print('\nPredicted questions:')
    print(sequences_to_text(train_token, ae_answer[rand]))

    print('\nFirst 10 answers:')
    print(sequences_to_text(train_token, test_t[rand]))
    print('\nPredicted answers:')
    print(sequences_to_text(train_token, qa_answer[rand]))

if __name__ == "__main__":
    sys.exit(main())
