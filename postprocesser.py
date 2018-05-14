from keras.backend import argmax
from tensorflow import InteractiveSession
from nltk.corpus import wordnet as wn
from tqdm import tqdm
import numpy as np

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

def sequences_to_text(token, x):
    print('Converting to text...')
    reverse_word_dict = dict(map(reversed, token.word_index.items()))
    InteractiveSession()
    seqs_to_words = lambda y: list(map(reverse_word_dict.get, y))
    words_to_sentence = lambda y: ' '.join(filter(None, y))
    sentence_list = list()
    for i in tqdm(range(len(x))):
        sentence_list.append(words_to_sentence(seqs_to_words(x[i])))
    return sentence_list

def sequences_to_matrix(token, x):
    print('Converting to text...')
    reverse_word_dict = dict(map(reversed, token.word_index.items()))
    InteractiveSession()
    seqs_to_words = lambda y: list(map(reverse_word_dict.get, y))
    word_matrix = list()
    for i in tqdm(range(len(x))):
        word_matrix.append(seqs_to_words(x[i]))
    return word_matrix  

def catsequences_to_matrix(token, x):
    print('Converting to text...')
    reverse_word_dict = dict(map(reversed, token.word_index.items()))
    InteractiveSession()
    from_categorical = lambda y: argmax(y, axis=-1).eval()
    seqs_to_words = lambda y: list(map(reverse_word_dict.get, from_categorical(y)))
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

def print_compare(questions,answers,predictions,qpredictions,N,token,compute_wups):
    rand = np.random.choice(5000, N)
    if (compute_wups):
        print('Converting Questions')
        questions = sequences_to_text(token,np.asarray(questions).tolist())
        print('Converting Answers')
        answers = sequences_to_text(token, answers.tolist())
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
        if not qpredictions is None:
            print('Converting predicted questions')
            qpredictions = catsequences_to_text(token,np.asarray(qpredictions)[rand].tolist())
        print('Converting answers')
        answers = sequences_to_matrix(token, answers[rand].tolist())
        print('Converting predicted answers')
        predictions = catsequences_to_matrix(token, predictions[rand].tolist())

    print('')
    if not qpredictions is None:
        acc1_q,acc2_q = accuracy_questions(questions,qpredictions)
        print('Accuracy1 Questions: ' + str(acc1_q))
        print('Accuracy2 Questions: ' + str(acc2_q))
    print_questions(questions,qpredictions)
    acc1_a,acc2_a = accuracy_answers(answers,predictions)
    print('Accuracy1 Answers: ' + str(acc1_a))
    print('Accuracy2 Answers: ' + str(acc2_a))
    print_answers(answers,predictions)

def accuracy_answers(answers,predictions):
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
    return acc1,acc2

def accuracy_questions(questions,qpredictions):
    acc1_num = 0
    acc1_denom = 0
    acc2_num = 0
    acc2_denom = 0
    for i in range(0,len(questions)):
        nb_words = len(questions[i].split(' '))
        nb_qwords = len(qpredictions[i].split(' '))
        nb_right = 0
        for j in range(min(nb_words,nb_qwords)):
            if questions[i].split(' ')[j] == qpredictions[i].split(' ')[j]:
                nb_right += 1
        acc1_num += nb_right/nb_words
        acc1_denom += 1
        acc2_num += nb_right
        acc2_denom += nb_words
    acc1 = acc1_num/acc1_denom
    acc2 = acc2_num/acc2_denom
    return acc1,acc2

def print_questions(questions,qpredictions):
    if not qpredictions is None:
        maxq = max(list(map(len,questions)))
        maxqp = max(list(map(len,qpredictions)))
    for i in range(0,len(questions)):
        if qpredictions is None:
            print(str(i)+'. '+questions[i])
        else:
            nb_words = len(questions[i].split(' '))
            nb_qwords = len(qpredictions[i].split(' '))
            nb_right = 0
            for j in range(min(nb_words,nb_qwords)):
                if questions[i].split(' ')[j] == qpredictions[i].split(' ')[j]:
                    nb_right += 1
            start = ' ' if i < 10 else ''
            correct = '+++ ' if nb_words == nb_right+1 else '--- '
            print(start + str(i)+'. '
                + ('{:'+str(maxq)+'} ').format(questions[i])
                + correct 
                + ('{:'+str(maxqp)+'} ').format(qpredictions[i])
                + str(nb_right)
                + '/'
                + str(nb_words))
    print('\n')

def print_answers(answers,predictions):
    maxa = list(map(lambda x: 0 if x is None else len(x),answers[0]))
    maxp = list(map(lambda x: 0 if x is None else len(x),predictions[0]))
    for i in range(0,len(answers)):
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
    for i in range(0,len(answers)):
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
