from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model
import numpy as np

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

def match_img_features(questions,img_features):
    try:
        imgs = np.asarray(list(map(lambda x: 
                    img_features[int(x.split('image')[-1].split(' ')[0])-1],
                    questions)))
    except ValueError:
        raise ValueError("Error matching images with questions, please use imageXXXX as the last word in the question")  
    return imgs
                
def preprocess(text, token, maxlen):
    text = token.texts_to_sequences(text)
    text = pad_sequences(text,maxlen=maxlen,padding='post')
    cat_text = to_categorical(text, len(token.word_index.items())+1)
    (N, sequence, voc) = cat_text.shape
    return text, cat_text, N, sequence, voc

def load_dataset(filename, k=0, token=None,img_filename=None,one_question=None):  
    if one_question is None:
        corpus = open(filename).read().lower().splitlines()
    else:
        corpus = [one_question.lower(),"None"]
    if not img_filename is None:
        img_features = load_cnn(img_filename)
        questions = corpus[0:2*k:2]
        imgs = match_img_features(questions,img_features)
    if token is None:
        token = Tokenizer(oov_token='~')#,filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~')
        token.fit_on_texts(corpus)
    q_corpus,q_cat_corpus, N, sequence, voc = preprocess(corpus, token, 30)
    a_corpus,a_cat_corpus, _, _, _ = preprocess(corpus, token, 11)
    # Extracting Training data and initializing some variables for the model
    x = q_corpus[0:2*k:2]  # extract every second item from the list
    t = a_corpus[1:2*k:2]
    x_cat = q_cat_corpus[0:2*k:2]  # extract every second item from the list
    t_cat = a_cat_corpus[1:2*k:2]
    #t = np.asarray(list(map(multiple_hot,t)))
    return x,x_cat,imgs,t,t_cat, N, sequence, voc, token