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
import numpy as np
import os.path
import argparse
import sys
import time

def load_cnn(filename):
    # Returns a matrix where each row corresponds to imageXXXX, 
    # note that index determines image number
    images = open(filename).read().splitlines()
    for i,img in enumerate(images):
        images[i] = img.split(',')
        images[i][0] = int(images[i][0][5:])
    images = np.asarray(images).astype(float)
    images = images[images[:,0].argsort()]
    return images[:,1:]

# def compute_wups(questions,answers,predictions,token):
def wup_measure(a,b,similarity_threshold=0.9):
    # Fetched from https://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/calculate_wups.py

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

    final_score=global_max*weight_a*weight_b*interp_weight*global_weight
    return final_score 

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

    # questions = sequences_to_text(token, questions)
    # answers = answermatrix_to_text(token, answers.tolist())
    # predictions = matrix_to_text(token, predictions.tolist())
    # print('\nWUPS measure with threshold 0.9')
    # our_element_membership=lambda x,y: wup_measure(x,y)
    # our_set_membership= lambda x,A: fuzzy_set_membership_measure(x,A,our_element_membership)
    # score_list=[score_it(answer,prediction,our_set_membership)
    #                 for (answer,prediction) in zip(answers,predictions)]
    # final_score=float(sum(score_list))/float(len(score_list))
    # print(final_score)
    # questions = np.asarray(questions)[rand]
    # answers = np.asarray(answers)[rand]
    # predictions = np.asarray(predictions)[rand]

    # return questions,answers,predictions

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
    text = pad_sequences(text,maxlen = 11,padding='post')
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
        token = Tokenizer(oov_token='~')#,filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~')
        token.fit_on_texts(corpus)
    q_corpus, N, sequence, voc = q_preprocess(corpus, token)
    a_corpus, _, _, _ = a_preprocess(corpus, token)
    # Extracting Training data and initializing some variables for the model
    x = q_corpus[0:2*k:2]  # extract every second item from the list
    t = a_corpus[1:2*k:2]
    #t = np.asarray(list(map(multiple_hot,t)))
    return x, imgs,t, N, sequence, voc, token

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
        name='decoder'
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

def sequences_to_text(token, x):
    print('Converting to text...')
    reverse_word_dict = dict(map(reversed, token.word_index.items()))
    InteractiveSession()
    from_categorical = lambda y: argmax(y, axis=-1).eval()
    seqs_to_words = lambda y: list(map(reverse_word_dict.get, y))
    words_to_sentence = lambda y: ' '.join(filter(None, y))
    # word_matrix = list(map(seqs_to_words, x))
    # sentence_list = list(map(words_to_sentence, word_matrix))
    sentence_list = list()
    for i in tqdm(range(len(x))):
        sentence_list.append(words_to_sentence(seqs_to_words(x[i])))
    return sentence_list

def catsequences_to_matrix(token, x):
    print('Converting to text...')
    reverse_word_dict = dict(map(reversed, token.word_index.items()))
    InteractiveSession()
    from_categorical = lambda y: argmax(y, axis=-1).eval()
    seqs_to_words = lambda y: list(map(reverse_word_dict.get, from_categorical(y)))
    # words_to_sentence = lambda y: ' '.join(filter(None, y))
    # word_matrix = list(map(seqs_to_words, x))
    # sentence_list = list(map(words_to_sentence, word_matrix))
    word_matrix = list()
    for i in tqdm(range(len(x))):
        word_matrix.append(seqs_to_words(x[i]))
    return word_matrix  

def catsequences_to_text(token, x):
    print('Converting to text...')
    reverse_word_dict = dict(map(reversed, token.word_index.items()))
    InteractiveSession()
    from_categorical = lambda y: argmax(y, axis=-1).eval()
    seqs_to_words = lambda y: list(map(reverse_word_dict.get, from_categorical(y)))
    words_to_sentence = lambda y: ' '.join(filter(None, y))
    # word_matrix = list(map(seqs_to_words, x))
    # sentence_list = list(map(words_to_sentence, word_matrix))
    sentence_list = list()
    for i in tqdm(range(len(x))):
        sentence_list.append(words_to_sentence(seqs_to_words(x[i])))
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

def answermatrix_to_text(token, x):
    print('Converting to text vector...')
    reverse_word_dict = dict(map(reversed, token.word_index.items()))
    InteractiveSession()
    seqs_to_words = lambda y: list(map(reverse_word_dict.get, argmax(y,axis=-1).eval()))
    y = list()
    for i in range(0,len(x)):
        a = x[i]
        b = list()
        for j in range(0,11):
            index = argmax(a,axis=-1).eval()
            answer = reverse_word_dict.get(index)
            a[index] = 0
            b.append(answer)
        y.append(b)
    # return seqs_to_words(x)
    return y

def print_compare(questions,answers,predictions,N,token,compute_wups):

    rand = np.random.choice(4000, N)
    if (compute_wups):
        print('Converting Questions')
        questions = sequences_to_text(token,np.asarray(questions).tolist())
        print('Converting Answers')
        answers = catsequences_to_text(token, answers.tolist())
        print('Converting Predictions')
        predictions = catsequences_to_text(token, predictions.tolist())
        print('\nWUPS measure with threshold 0.9')
        our_element_membership=lambda x,y: wup_measure(x,y)
        our_set_membership= lambda x,A: fuzzy_set_membership_measure(x,A,our_element_membership)
        score_list = list()
        for i in tqdm(range(0,len(answers))):
            score_list.append(score_it(answers[i],predictions[i],our_set_membership))
        # score_list=[score_it(answer,prediction,our_set_membership)
        #                 for (answer,prediction) in zip(answers,predictions)]
        final_score=float(sum(score_list))/float(len(score_list))
        print(final_score)
        # questions = np.asarray(questions)[rand]
        # answers = np.asarray(answers)[rand]
        # predictions = np.asarray(predictions)[rand]
    else:
        print('Converting questions')
        questions = sequences_to_text(token,np.asarray(questions)[rand].tolist())
        print('Converting answers')
        answers = catsequences_to_matrix(token, answers[rand].tolist())
        print('Converting predictions')
        predictions = catsequences_to_matrix(token, predictions[rand].tolist())

    print('\n')
    for i in range(0,N):
        print(str(i)+'. '+questions[i])
    print('\n')
    # for i in range(0,N):
    #     start = ' ' if i < 10 else ''
    #     print(start + str(i)+'. ' 
    #         + answers[i]
    #         + ' --- '
    #         + predictions[i])

    acc1_num = 0
    acc1_denom = 0
    acc2_num = 0
    acc2_denom = 0
    for i in range(0,len(answers)):
        nb_answers = len(set(filter(lambda x: not x is None,answers[i])))
        nb_right = len(set(filter(lambda x: not x is None,answers[i])).intersection(predictions[i]))
        acc1_num += nb_right/nb_answers
        acc1_denom += 1
        acc2_num += nb_right
        acc2_denom += nb_answers
    acc1 = acc1_num/acc1_denom
    acc2 = acc2_num/acc2_denom
    print('Accuracy1: ' + str(acc1_num) + '/' + str(acc1_denom) + ' = ' + str(acc1))
    print('Accuracy2: ' + str(acc2_num) + '/' + str(acc2_denom) + ' = ' + str(acc2))

    maxa = list(map(lambda x: 0 if x is None else len(x),answers[0]))
    maxp = list(map(lambda x: 0 if x is None else len(x),predictions[0]))
    for i in range(0,N):
        for j in range(0,len(answers[0])):
            if not answers[i][j] is None and len(answers[i][j]) > maxa[j]:
                maxa[j] = len(answers[i][j])
            if not predictions[i][j] is None and len(predictions[i][j]) > maxp[j]:
                maxp[j] = len(predictions[i][j])
    formata = ''
    formatp = ''
    for i in range(0,len(answers[0])):
        if maxa[i] > 0:
            formata += '{:'+str(maxa[i])+'} '
        if maxp[i] > 0:
            formatp += '{:'+str(maxp[i])+'} '
    for i in range(0,N):
        nb_answers = len(set(filter(lambda x: not x is None,answers[i])))
        nb_right = len(set(filter(lambda x: not x is None,answers[i])).intersection(predictions[i]))
        start = ' ' if i < 10 else ''
        correct = ' +++ ' if nb_right > 0 else ' --- '
        answerlist = list(map(lambda x: "" if x is None else x,answers[i]))
        predictionlist = list(map(lambda x: "" if x is None else x,predictions[i]))
        print(start + str(i) + '. '
             + formata.format(*answerlist)
             + correct
             + formatp.format(*predictionlist)
             + str(nb_right)
             + '/'
             + str(nb_answers))

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
    train_x, train_imgs, train_t, train_N, train_sequence, voc, train_token = load_dataset("qa.894.raw.train.txt", 6795,img_filename="img_features.csv")
    test_x, test_imgs, test_t, test_N, test_sequence, _, _ = load_dataset("qa.894.raw.test.txt", 6795, train_token, img_filename="img_features.csv")


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

    print_compare(test_x,test_t,qa_answer,100,train_token,args.wups)


if __name__ == "__main__":
    sys.exit(main())
