# Originally adapted from https://github.com/keras-team/keras/issues/1401
# https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
# Requires Keras and Tensorflow backend

from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Embedding, Input, Dropout, concatenate
from keras.layers.recurrent import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.backend import argmax
from keras.callbacks import ModelCheckpoint
from tensorflow import InteractiveSession
from keras import regularizers
from nltk.corpus import wordnet as wn
import numpy as np
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


def wup_measure(a,b,similarity_threshold=0.9):
    # Fetched from https://www.programcreek.com/python/example/91610/nltk.corpus.wordnet.NOUN
    # Original Author: mateuszmalinowski


    """
    Returns Wu-Palmer similarity score.
    More specifically, it computes:
        max_{x \in interp(a)} max_{y \in interp(b)} wup(x,y)
        where interp is a 'interpretation field'
    """
    def get_semantic_field(a):
        weight = 1.0
        semantic_field = wn.synsets(a,pos=wn.NOUN)
        return (semantic_field,weight)


    def get_stem_word(a):
        """
        Sometimes answer has form word\d+:wordid.
        If so we return word and downweight
        """
        weight = 1.0
        return (a,weight)


    global_weight=1.0

    (a,global_weight_a)=get_stem_word(a)
    (b,global_weight_b)=get_stem_word(b)
    global_weight = min(global_weight_a,global_weight_b)

    if a==b:
        # they are the same
        return 1.0*global_weight

    if a==[] or b==[]:
        return 0


    interp_a,weight_a = get_semantic_field(a) 
    interp_b,weight_b = get_semantic_field(b)

    if interp_a == [] or interp_b == []:
        return 0

    # we take the most optimistic interpretation
    global_max=0.0
    for x in interp_a:
        for y in interp_b:
            local_score=x.wup_similarity(y)
            if local_score > global_max:
                global_max=local_score

    # we need to use the semantic fields and therefore we downweight
    # unless the score is high which indicates both are synonyms
    if global_max < similarity_threshold:
        interp_weight = 0.1
    else:
        interp_weight = 1.0

    final_score = global_max * weight_a * weight_b * interp_weight * global_weight
    return final_score 

    
def match_img_features(questions,img_features):
    return np.asarray(list(map(lambda x: 
                img_features[int(x.split('image')[-1].split(' ')[0])-1],
                questions)))

                
def preprocess(text, token):
    text = token.texts_to_sequences(text)
    text = pad_sequences(
        text,
        maxlen = 30)
    # text = to_categorical(text, len(token.word_index.items())+1)
    (N, sequence) = text.shape
    voc = len(token.word_index.items())+1
    return text, N, sequence, voc

def q_preprocess(text, token):
    text = token.texts_to_sequences(text)
    text = pad_sequences(text,maxlen = 30)
    # text = to_categorical(text, len(token.word_index.items())+1)
    (N, sequence) = text.shape
    voc = len(token.word_index.items())+1
    return text, N, sequence, voc

def a_preprocess(text, token):
    text = token.texts_to_sequences(text)
    text = pad_sequences(text,maxlen = 30)
    text = to_categorical(text, len(token.word_index.items())+1)
    (N, sequence,voc) = text.shape
    return text, N, sequence, voc

def multiple_hot(sequence):
    sum = sequence[0]
    for i in range(1,len(sequence)):
            sum += sequence[i]
    sum[0] = 0
    return sum

def load_dataset(filename, k=0, token=None,img_filename=None):
    corpus = open(filename).read().lower().splitlines()
    if not img_filename is None:
        img_features = load_cnn(img_filename)
        questions = corpus[0:2*k:2]
        imgs = match_img_features(questions,img_features)
    if token is None:
        token = Tokenizer(oov_token='~')
        token.fit_on_texts(corpus)
    q_corpus, N, sequence, voc = q_preprocess(corpus, token)
    a_corpus, _, _, _ = a_preprocess(corpus, token)
    # Extracting Training data and initializing some variables for the model
    x = q_corpus[0:2*k:2]  # extract every second item from the list
    t = a_corpus[1:2*k:2]
    t = np.asarray(list(map(multiple_hot,t)))
    print(x.shape)
    print(t.shape)
    return x, imgs,t, N, sequence, voc, token


def build_classifier(words, images, t, e,l1, voc, batch):


    # Build text based input. embedding and LSTM layer
    # The Input layer is only there for convenience and doesn't do anything
    # The Embedding layer takes a 2D input (batch_size, sequence_length) and
    # outputs a 3D tensor (batch_size,sequence_length,embedding_dim) for the 
    # LSTM. The LSTM and Embedding layer match dimensions for convience. The
    # LSTM is configured to only pass information forward at the end of the 
    # sequence. 
    word_input = Input(shape=(30,))
    word_embedding = Embedding(
        input_dim = voc,
        output_dim = l1,
        mask_zero = True
        )(word_input)
    word_encoding = LSTM(
        l1,
        return_sequences = False)(word_embedding) 


    # Construtct the Image input part. Since no feature extraction 
    # takes place we basically just run ahead here
    visual_input = Input(shape=(images.shape[1],))
    visual_encoding = Dense(images.shape[1])(visual_input) 



    # We merge the model, add a dropout to combat some overfitting and fit.
    merged = concatenate([word_encoding,visual_encoding])
    dropout = Dropout(0.5)(merged)
    output = Dense(
        voc,
        activation = 'softmax',
        name = 'Output_layer')(dropout)
    classifier = Model(
        inputs = [word_input,visual_input], 
        outputs = output)
    classifier.compile(
        loss = 'categorical_crossentropy', 
        optimizer = 'adam', 
        metrics = ['categorical_accuracy'])
    print('Training...')
    classifier.fit(
        [words, images],
        t, 
        epochs = e, 
        batch_size = batch,
        validation_split = 0.1)
    return classifier


def sequences_to_text(token, x):
    print('Converting to text...')
    reverse_word_dict = dict(map(reversed, token.word_index.items()))
    InteractiveSession()
    from_categorical = lambda y: argmax(y, axis=-1).eval()
    seqs_to_words = lambda y: list(map(reverse_word_dict.get, y))
    words_to_sentence = lambda y: ' '.join(filter(None, y))
    word_matrix = list(map(seqs_to_words, x))
    sentence_list = list(map(words_to_sentence, word_matrix))
    return sentence_list

def sequence_to_text(token, x):
    print('Converting to text...')
    reverse_word_dict = dict(map(reversed, token.word_index.items()))
    InteractiveSession()
    from_categorical = lambda y: argmax(y, axis=-1).eval()
    return reverse_word_dict.get(from_categorical(x))

def matrix_to_text(token, x):
    print('Converting to text vector...')
    reverse_word_dict = dict(map(reversed, token.word_index.items()))
    InteractiveSession()
    seqs_to_words = lambda y: list(map(reverse_word_dict.get, argmax(y,axis=-1).eval()))
    return seqs_to_words(x)

def fuzzy_set_membership_measure(x,A,m):
    """
    Set membership measure.
    x: element
    A: set of elements
    m: point-wise element-to-element measure m(a,b) ~ similarity(a,b)

    This function implments a fuzzy set membership measure:
        m(x \in A) = max_{a \in A} m(x,a)}
    """
    return 0 if A==[] else max(list(map(lambda a: m(x,a), A)))

def score_it(A,T,m):
    """
    A: list of A items 
    T: list of T items
    m: set membership measure
        m(a \in A) gives a membership quality of a into A 

    This function implements a fuzzy accuracy score:
        score(A,T) = min{prod_{a \in A} m(a \in T), prod_{t \in T} m(a \in A)}
        where A and T are set representations of the answers
        and m is a measure
    """
    if A==[] and T==[]:
        return 1

    # print A,T

    score_left=0 if A==[] else np.prod(list(map(lambda a: m(a,T), A)))
    score_right=0 if T==[] else np.prod(list(map(lambda t: m(t,A),T)))
    return min(score_left,score_right) 


def print_compare(questions,answers,predictions,N,token,compute_wups):
    rand = np.random.choice(4000, N)
    # questions = sequences_to_text(token, questions)
    # answers = matrix_to_text(token, answers.tolist())
    # predictions = matrix_to_text(token, predictions.tolist())
    if (compute_wups):
        questions = sequences_to_text(token, questions)
        answers = matrix_to_text(token, answers.tolist())
        predictions = matrix_to_text(token, predictions.tolist())
        print('\nWUPS measure with threshold 0.9')
        our_element_membership=lambda x,y: wup_measure(x,y)
        our_set_membership= lambda x,A: fuzzy_set_membership_measure(x,A,our_element_membership)
        score_list=[score_it(answer,prediction,our_set_membership)
                        for (answer,prediction) in zip(answers,predictions)]
        final_score=float(sum(score_list))/float(len(score_list))
        print(final_score)
        questions = np.asarray(questions)[rand]
        answers = np.asarray(answers)[rand]
        predictions = np.asarray(predictions)[rand]
    else:
        questions = sequences_to_text(token,np.asarray(questions)[rand].tolist())
        answers = matrix_to_text(token, answers[rand].tolist())
        predictions = matrix_to_text(token, predictions[rand].tolist())
    print('\n')
    for i in range(0,N):
        print(str(i)+'. '+questions[i])
    print('\n')
    print('    Real' + '\t --- \t' + 'Prediction')
    for i in range(0,N):
        correct = '+++' if answers[i] == predictions[i] else '---'
        mid = '\t\t ' + correct + ' \t' if len(answers[i]) < 4 else '\t ' + correct + ' \t'
        start = ' ' if i < 10 else ''
        print(start + str(i) + '. ' + answers[i] + mid + predictions[i])

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
                           help='Batch size, default 32')
    argparser.add_argument('--wups', action="store_true",
                           help='Compute the WUPS Score')
    args = argparser.parse_args(argv)


    # Hyper Parameters
    ld1 = 512
    epochs = 20
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

    qa_file_id = 'Question_Answerer_' + str(epochs) + '_' + str(ld1) + '.h5'

    # This function fetches the dataset from the file and fills both X and T with k number of datapoints
    train_x, train_imgs, train_t, train_N, train_sequence, voc, train_token = load_dataset("qa.894.raw.train.txt", 6795,img_filename="img_features.csv")
    test_x, test_imgs, test_t, test_N, test_sequence, _, _ = load_dataset("qa.894.raw.test.txt", 6795, train_token, img_filename="img_features.csv")
    print(train_x.shape)
    print(train_t.shape)


    # Build and train Question Answerer
    if args.qa:
        print('\nLoading Question Answerer model from file: ' + args.qa + ' \n')
        classifier = load_model(args.qa)
    elif os.path.exists(qa_file_id):
        print('\nQuestion Answerer model with these parameters found, loading model from file: ' + qa_file_id + '\n')
        classifier = load_model(qa_file_id)
    else:
        print('\nNo question answerer model with these parameters was found, building new model.\n')
        classifier = build_classifier(train_x, train_imgs, train_t, epochs, ld1, voc, batch)
        classifier.save(qa_file_id)
        print('\nModel saved to: ' + qa_file_id)

    

    print('\nEvaluating question answerer on test data')
    qa_result = classifier.evaluate([test_x,test_imgs], test_t, batch_size = batch)
    qa_answer = classifier.predict([test_x,test_imgs], batch_size = batch)
    print('Loss: ' + str(qa_result[0]))
    print('Accuracy: ' + str(qa_result[1]))

    print_compare(test_x,test_t,qa_answer,20,train_token,args.wups)



    

if __name__ == "__main__":
    sys.exit(main())
