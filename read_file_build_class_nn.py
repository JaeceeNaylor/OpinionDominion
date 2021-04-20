import csv
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
import stanza
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
#from sklearn_crfsuite import CRF
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import re
import scoring
import output_file_creation
import parsing
import review_opinion_sent
import sys
import re
import tensorflow as tf
import keras
from keras import Sequential, Model, Input
from keras.layers import Dense, Embedding, GRU, Dropout, Bidirectional, SpatialDropout1D, Activation, LSTM, TimeDistributed, Dropout
from keras.utils import to_categorical, plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from numpy.random import seed

seed(1)
tf.random.set_seed(2)
max_features = 4000


nltk.download('stopwords')

    
def process_reviews(all_reviews, stop_words):
    end_vocab, all_vocab, all_pos, all_word_shapes, all_word_types = create_vocab(all_reviews, stop_words)
    '''
    print(f'VOCAB: {end_vocab}')
    print()
    '''
    X_Category, Y_Category, X_Polarity, Y_Polarity = create_feature_vectors_and_expected_values(all_reviews, end_vocab)
    X_Target_BIO, Y_Target_BIO, maxlen = create_feature_vectors_bio(all_reviews, all_vocab, all_pos, all_word_shapes, all_word_types)
    X_C_Train, X_C_Test, Y_C_Train, Y_C_Test = train_test_split(X_Category, Y_Category, random_state = 0, shuffle = False) #default 25% become test examples
    X_P_Train, X_P_Test, Y_P_train, Y_P_Test = train_test_split(X_Polarity, Y_Polarity, random_state = 0, shuffle = False)
    X_T_Train, X_T_Test, Y_T_Train, Y_T_Test = train_test_split(X_Target_BIO, Y_Target_BIO, random_state = 0, shuffle = False)

    clf_l = LogisticRegression(random_state=0).fit(X_C_Train, Y_C_Train)
    clf_l_category_predictions = clf_l.predict(X_C_Test)
    accuracy = clf_l.score(X_C_Test, Y_C_Test)
    print(f'CATEGORY ACCURACY Linear Regresion: {accuracy}')
    l_model_polarty = LogisticRegression(random_state=0).fit(X_P_Train, Y_P_train)
    l_polarity_predications = l_model_polarty.predict(X_P_Test)
    accuracy = l_model_polarty.score(X_P_Test, Y_P_Test)
    print(f'POLARITY ACCURACY Linear Regresion: {accuracy}')
    input_dim = len(all_vocab)+1
    output_dim = 64
    input_length = maxlen
    n_tags = 2
    model = get_model(input_dim, output_dim, input_length, n_tags, X_T_Train)
    train_model(X_T_Train, np.array(Y_T_Train), model)
    model.save('myModel')
    crf_target_predictions = np.argmax(model.predict(X_T_Test), axis=-1) #model.predict_classes(X_T_Test)
    print(crf_target_predictions)
    print()

    predict_opinions(all_reviews, X_C_Train, Y_Target_BIO, clf_l_category_predictions, l_polarity_predications, crf_target_predictions)
    scoring.calculate_scores(all_reviews)

    #cross validation
    kf = KFold(n_splits = 5)
    kf.get_n_splits(X_Category)
    kf.get_n_splits(X_Polarity)
    kf.get_n_splits(X_Target_BIO)
    fold = 1
    for train_index, test_index in kf.split(X_Category):
        print()
        print(f'FOLD: {fold}')
        fold += 1
        X_C_Train, X_C_Test = X_Category[train_index], X_Category[test_index]
        Y_C_Train, Y_C_Test = Y_Category[train_index], Y_Category[test_index]

        X_P_Train, X_P_Test = X_Polarity[train_index], X_Polarity[test_index]
        Y_P_train, Y_P_Test = Y_Polarity[train_index], Y_Polarity[test_index]

        X_T_Train = []
        X_T_Test = []
        Y_T_Train = []
        Y_T_Test = []
        for t_i in train_index:
            X_T_Train.append(X_Target_BIO[t_i])
            Y_T_Train.append(Y_Target_BIO[t_i])
        for t_i in test_index:
            X_T_Test.append(X_Target_BIO[t_i])
            Y_T_Test.append(Y_Target_BIO[t_i])

        clf_l = LogisticRegression(random_state=0).fit(X_C_Train, Y_C_Train)
        clf_l_category_predictions = clf_l.predict(X_C_Test)
        accuracy = clf_l.score(X_C_Test, Y_C_Test)
        print(f'CATEGORY ACCURACY Linear Regresion: {accuracy}')
        l_model_polarty = LogisticRegression(random_state=0).fit(X_P_Train, Y_P_train)
        l_polarity_predications = l_model_polarty.predict(X_P_Test)
        accuracy = l_model_polarty.score(X_P_Test, Y_P_Test)
        print(f'POLARITY ACCURACY Linear Regresion: {accuracy}')
        model = get_model(input_dim, output_dim, input_length, n_tags, X_T_Train)
        train_model(np.array(X_T_Train), np.array(Y_T_Train), model)
        crf_target_predictions = np.argmax(model.predict(np.array(X_T_Test)), axis=-1)
        print(f'TARGET ACCURACY CRF: {1}')
        print()

        predict_opinions_cv(all_reviews, train_index, test_index, clf_l_category_predictions, l_polarity_predications, crf_target_predictions)
        scoring.calculate_scores(all_reviews)
    
    target_predicted_file_name = 'output_target_data/trial_CRF.target.predicted'
    target_expected_file_name = 'output_target_data/trial_CRF.target.expected'
    output_file_creation.create_files(target_predicted_file_name, target_expected_file_name, all_reviews, 'TARGET')

    polarity_predicted_file_name = 'output_polarity_data/trial_CRF.target.predicted'
    polarity_expected_file_name = 'output_polarity_data/trial_CRF.target.expected'
    output_file_creation.create_files(polarity_predicted_file_name, polarity_expected_file_name, all_reviews, 'POLARITY')

    category_predicted_file_name = 'output_category_data/trial_CRF.target.predicted'
    category_expected_file_name = 'output_category_data/trial_CRF.target.expected'
    output_file_creation.create_files(category_predicted_file_name, category_expected_file_name, all_reviews, 'CATEGORY')

def process_reviews_on_predetermined_model(all_reviews, stop_words, model):
    end_vocab, all_vocab, all_pos, all_word_shapes, all_word_types = create_vocab(all_reviews, stop_words)
    X_Category, Y_Category, X_Polarity, Y_Polarity = create_feature_vectors_and_expected_values(all_reviews, end_vocab)
    X_Target_BIO, Y_Target_BIO, maxlen = create_feature_vectors_bio(all_reviews, all_vocab, all_pos, all_word_shapes, all_word_types)
    X_C_Train, X_C_Test, Y_C_Train, Y_C_Test = train_test_split(X_Category, Y_Category, random_state = 0, shuffle = False) #default 25% become test examples
    X_P_Train, X_P_Test, Y_P_train, Y_P_Test = train_test_split(X_Polarity, Y_Polarity, random_state = 0, shuffle = False)
    X_T_Train, X_T_Test, Y_T_Train, Y_T_Test = train_test_split(X_Target_BIO, Y_Target_BIO, random_state = 0, shuffle = False)

    clf_l = LogisticRegression(random_state=0).fit(X_C_Train, Y_C_Train)
    clf_l_category_predictions = clf_l.predict(X_C_Test)
    accuracy = clf_l.score(X_C_Test, Y_C_Test)
    l_model_polarty = LogisticRegression(random_state=0).fit(X_P_Train, Y_P_train)
    l_polarity_predications = l_model_polarty.predict(X_P_Test)
    accuracy = l_model_polarty.score(X_P_Test, Y_P_Test)
    crf_target_predictions = np.argmax(model.predict(X_T_Test), axis=-1) #model.predict_classes(X_T_Test)
    print(crf_target_predictions)
    print()

    predict_opinions(all_reviews, X_C_Train, Y_Target_BIO, clf_l_category_predictions, l_polarity_predications, crf_target_predictions)
    
    target_predicted_file_name = 'output_target_data/trial_CRF.target.predicted'
    target_expected_file_name = 'output_target_data/trial_CRF.target.expected'
    output_file_creation.create_files(target_predicted_file_name, target_expected_file_name, all_reviews, 'TARGET')

    polarity_predicted_file_name = 'output_polarity_data/trial_CRF.target.predicted'
    polarity_expected_file_name = 'output_polarity_data/trial_CRF.target.expected'
    output_file_creation.create_files(polarity_predicted_file_name, polarity_expected_file_name, all_reviews, 'POLARITY')

    category_predicted_file_name = 'output_category_data/trial_CRF.target.predicted'
    category_expected_file_name = 'output_category_data/trial_CRF.target.expected'
    output_file_creation.create_files(category_predicted_file_name, category_expected_file_name, all_reviews, 'CATEGORY')

def get_model(input_dim, output_dim, input_length, n_tags, padded_inputs):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length, mask_zero=True))
    #print(padded_inputs[0])
    #masked_output = model(padded_inputs)
    #print(masked_output._keras_mask)
    model.add(Bidirectional(LSTM(units=output_dim, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), merge_mode = 'concat'))
    
    #model.add(Bidirectional(GRU(units=output_dim, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), merge_mode = 'concat'))
    #model.add(LSTM(units=output_dim, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
    model.add(TimeDistributed(Dense(n_tags, activation="softmax")))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.summary()
    return model


def train_model(X, y, model):
    #print(y.shape)
    #print(X.shape)
    loss = list()
    for i in range(25):
        hist = model.fit(X, y, batch_size=100, verbose=1, epochs=1, validation_split=0.2)
        loss.append(hist.history['loss'][0])
    return loss

def process_reviews_all(all_reviews, all_test_reviews, stop_words):
    end_vocab, all_vocab, all_pos, all_word_shapes, all_word_types = create_vocab(all_reviews, stop_words)
    end_t_vocab, all_t_vocab, all_t_pos, all_t_word_shapes, all_t_word_types = create_vocab(all_test_reviews, stop_words)
    for key in all_t_vocab:
        if key not in all_vocab:
            all_vocab[key] = 1
    for key in all_t_pos:
        if key not in all_pos:
            all_pos[key] = 1
    for key in all_t_word_shapes:
        if key not in all_word_shapes:
            all_word_shapes[key] = 1
    for key in all_t_word_types:
        if key not in all_word_types:
            all_word_types[key] = 1
    X_C_Train, Y_C_Train, X_P_Train, Y_P_train = create_feature_vectors_and_expected_values(all_reviews, end_vocab)
    X_T_Train, Y_T_Train, maxlen = create_feature_vectors_bio(all_reviews, all_vocab, all_pos, all_word_shapes, all_word_types)
    X_C_Test, Y_C_Test, X_P_Test, Y_P_Test = create_feature_vectors_and_expected_values(all_test_reviews, end_vocab)
    X_T_Test, Y_T_Test, maxlen = create_feature_vectors_bio(all_test_reviews, all_vocab, all_pos, all_word_shapes, all_word_types)

    clf_l = LogisticRegression(random_state=0).fit(X_C_Train, Y_C_Train)
    clf_l_category_predictions = clf_l.predict(X_C_Test)
    accuracy = clf_l.score(X_C_Test, Y_C_Test)
    print(f'CATEGORY ACCURACY Linear Regresion: {accuracy}')
    l_model_polarty = LogisticRegression(random_state=0).fit(X_P_Train, Y_P_train)
    l_polarity_predications = l_model_polarty.predict(X_P_Test)
    accuracy = l_model_polarty.score(X_P_Test, Y_P_Test)
    print(f'POLARITY ACCURACY Linear Regresion: {accuracy}')
    input_dim = len(all_vocab)+1
    output_dim = 64
    input_length = maxlen
    n_tags = 2
    model = get_model(input_dim, output_dim, input_length, n_tags, X_T_Train)
    train_model(np.array(X_T_Train), np.array(Y_T_Train), model)
    crf_target_predictions = model.predict_classes(np.array(X_T_Test))
    print(f'TARGET ACCURACY CRF: {1}')
    print()

    predict_opinions_all(all_test_reviews, clf_l_category_predictions, l_polarity_predications, crf_target_predictions)
    scoring.calculate_scores(all_test_reviews)



def predict_opinions_cv(all_reviews, train_index, test_index, l_category_predictions, l_polarity_predications, crf_target_predictions):
    train_i_d = {}
    test_i_d = {}
    for t_index in train_index:
        train_i_d[t_index] = ''
    for t_index in test_index:
        test_i_d[t_index] = ''
    i = 0
    predict_index = 0
    category_to_ote = {}
    for review in all_reviews:
        for sentence in review.sentences:
            sentence.opinions_predicted = []
            for opinion in sentence.opinions_expected:
                if i in train_i_d:
                    if opinion.category in category_to_ote:
                        category_to_ote[opinion.category][opinion.target] = ''
                    elif opinion.category not in category_to_ote:
                        category_to_ote[opinion.category] = {opinion.target: ''}
                    i += 1
                    continue
                category_pred = l_category_predictions[predict_index]
                #category_pred = clf_category_predictions[predict_index]
                target = ''
                i += 1
                t_f = False
                for k in range(len(sentence.words)):
                    word = sentence.words[k]
                    if crf_target_predictions[predict_index][k] == 0 and t_f == False and (k == 0 or crf_target_predictions[predict_index][k-1] == 2):
                        t_f = True
                        if k < len(sentence.words) - 1 and crf_target_predictions[predict_index][k+1] == 1:
                            word += ' ' + sentence.words[k+1]
                            if k < len(sentence.words) - 2 and crf_target_predictions[predict_index][k+2] == 1:
                                word += ' ' + sentence.words[k+2]
                                if k < len(sentence.words) - 3 and crf_target_predictions[predict_index][k+3] == 1:
                                    word += ' ' + sentence.words[k+3]
                        target += word
                if target == '':
                    target = 'NULL'
                #classifier for each performs worse -- one vs rest is below this comment
                #opinion_predicted = review_opinion_sent.Opinion(sentence.sentence_id, sentence.review_id, target, '0', '0', clf_category_predictions[predict_index], clf_polarity_predictions[predict_index])
                opinion_predicted = review_opinion_sent.Opinion(sentence.sentence_id, sentence.review_id, target, '0', '0', l_category_predictions[predict_index], l_polarity_predications[predict_index])
                opinion_predicted.setPolarity(sentence)
                #opinion_predicted.print_attr()
                predict_index += 1
                sentence.opinions_predicted.append(opinion_predicted)

def predict_opinions_all(all_reviews, l_category_predictions, l_polarity_predications, crf_target_predictions):
    i = 0
    predict_index = 0
    category_to_ote = {}
    for review in all_reviews:
        for sentence in review.sentences:
            sentence.opinions_predicted = []
            for opinion in sentence.opinions_expected:
                target = ''
                '''
                print(crf_target_predictions[predict_index])
                print(Y_Target_BIO[i])
                '''
                i += 1
                t_f = False
                for k in range(len(sentence.words)):
                    word = sentence.words[k]
                    if crf_target_predictions[predict_index][k] == 0 and t_f == False and (k == 0 or crf_target_predictions[predict_index][k-1] == 2):
                        t_f = True
                        if k < len(sentence.words) - 1 and crf_target_predictions[predict_index][k+1] == 1:
                            word += ' ' + sentence.words[k+1]
                            if k < len(sentence.words) - 2 and crf_target_predictions[predict_index][k+2] == 1:
                                word += ' ' + sentence.words[k+2]
                                if k < len(sentence.words) - 3 and crf_target_predictions[predict_index][k+3] == 1:
                                    word += ' ' + sentence.words[k+3]
                        target += word
                if target == '':
                    target = 'NULL'
                #classifier for each performs worse -- one vs rest is below this comment
                #opinion_predicted = review_opinion_sent.Opinion(sentence.sentence_id, sentence.review_id, target, '0', '0', clf_category_predictions[predict_index], clf_polarity_predictions[predict_index])
                opinion_predicted = review_opinion_sent.Opinion(sentence.sentence_id, sentence.review_id, target, '0', '0', l_category_predictions[predict_index], l_polarity_predications[predict_index])
                opinion_predicted.setPolarity(sentence)
                #opinion_predicted.print_attr()
                predict_index += 1
                sentence.opinions_predicted.append(opinion_predicted)

def predict_opinions(all_reviews, X_C_Train, Y_Target_BIO, l_category_predictions, l_polarity_predications, crf_target_predictions):    
    i = 0
    predict_index = 0
    category_to_ote = {}
    for review in all_reviews:
        for sentence in review.sentences:
            sentence.opinions_predicted = []
            for opinion in sentence.opinions_expected:
                if i < len(X_C_Train):
                    if opinion.category in category_to_ote:
                        category_to_ote[opinion.category][opinion.target] = ''
                    elif opinion.category not in category_to_ote:
                        category_to_ote[opinion.category] = {opinion.target: ''}
                    i += 1
                    continue
                category_pred = l_category_predictions[predict_index]
                #category_pred = clf_category_predictions[predict_index]
                target = ''
                '''
                print(crf_target_predictions[predict_index])
                print(Y_Target_BIO[i])
                '''
                i += 1
                t_f = False
                for k in range(len(sentence.words)):
                    word = sentence.words[k]
                    #0 = B, 1 = I, 2 = O
                    '''
                    if crf_target_predictions[predict_index][k] == 0:
                        print('here 1')
                        print(k == 0)
                        print(t_f)
                        if k != 0:
                            print(crf_target_predictions[predict_index][k-1] == 1)
                    '''
                    if crf_target_predictions[predict_index][k] == 1 and t_f == False and (k == 0 or crf_target_predictions[predict_index][k-1] == 0):
                        t_f = True
                        print(f'WORD 1: {word}')
                        if k < len(sentence.words) - 1 and crf_target_predictions[predict_index][k+1] == 1:
                            print(f'WORD 2: {sentence.words[k+1]}')
                            word += ' ' + sentence.words[k+1]
                            if k < len(sentence.words) - 2 and crf_target_predictions[predict_index][k+2] == 1:
                                word += ' ' + sentence.words[k+2]
                                print(f'WORD 3: {sentence.words[k+2]}')
                                if k < len(sentence.words) - 3 and crf_target_predictions[predict_index][k+3] == 1:
                                    word += ' ' + sentence.words[k+3]
                        target += word
                if target == '':
                    target = 'NULL'
                #classifier for each performs worse -- one vs rest is below this comment
                #opinion_predicted = review_opinion_sent.Opinion(sentence.sentence_id, sentence.review_id, target, '0', '0', clf_category_predictions[predict_index], clf_polarity_predictions[predict_index])
                opinion_predicted = review_opinion_sent.Opinion(sentence.sentence_id, sentence.review_id, target, '0', '0', l_category_predictions[predict_index], l_polarity_predications[predict_index])
                opinion_predicted.setPolarity(sentence)
                #opinion_predicted.print_attr()
                predict_index += 1
                sentence.opinions_predicted.append(opinion_predicted)

#Could expand BIO to match on the specific categories as well
def create_feature_vectors_bio(all_reviews, all_vocab, all_pos, all_word_shapes, all_word_types):
    X_BIO = []
    Y_BIO = []
    for review in all_reviews:
        for sentence in review.sentences:
            for opinion in sentence.opinions_expected:
                answer = []
                word_feature_vec = []
                label_vec = []
                for i in range(len(sentence.words)):
                    
                    word = sentence.words[i]
                    word_feature_vec.append(all_vocab[word])
                    word_feature_vec.append(all_pos[sentence.pos_tags_on_words_in_sentence[i]])
                    word_feature_vec.append(all_word_shapes[sentence.word_shapes[i]])
                    word_feature_vec.append(all_word_types[sentence.word_types[i]])
                    target_words = opinion.target.split()
                    target_found = False
                    for j in range(len(target_words)):
                        if target_words[j] == word and j == 0:
                            #print('here 2')
                            target_found = True
                            label_vec.append(0) #'B'
                        elif target_words[j] == word and j >= 1 and target_found == False:
                            #print('here 1')
                            target_found = True
                            label_vec.append(0) #'I'
                    if target_found == False:
                        label_vec.append(1) #'O'
                X_BIO.append(word_feature_vec)
                Y_BIO.append(label_vec)
    #print(f'LEN: {len(all_vocab)}')
    n_token = len(all_vocab)
    n_tags = 2
    maxlen = max([len(s) for s in X_BIO])
    #print(f'X_BIO before: {X_BIO[0]}')
    X_BIO = pad_sequences(X_BIO, maxlen=maxlen, dtype='int32', padding='post')
    X_BIO = np.asarray(X_BIO)
    #print(f'X_BIO after: {X_BIO[0]}')
    Y_BIO = pad_sequences(Y_BIO, maxlen=maxlen, dtype='int32', padding='post')
    #print(f'Y_BIO : {Y_BIO[0]}')
    #print(f'X_BIO : {len(X_BIO[0])}')
    Y_BIO = [to_categorical(i, num_classes=n_tags) for i in Y_BIO]
    #print(f'Y_BIO : {Y_BIO[0]}')
    #print(f'Y_BIO : {Y_BIO}')
    Y_BIO = np.asarray(Y_BIO)
    return X_BIO, Y_BIO, maxlen
                        

def create_feature_vectors_and_expected_values(all_reviews, vocab):
    category_dict = {
        'RESTAURANT#GENERAL': 0, 'RESTAURANT#PRICES': 1, 'RESTAURANT#MISCELLANEOUS': 3, 
        'FOOD#PRICES': 4, 'FOOD#QUALITY': 5, 'FOOD#STYLE_OPTIONS': 6, 
        'DRINKS#PRICES': 7, 'DRINKS#QUALITY': 7, 'DRINKS#STYLE_OPTIONS': 8,    
        'AMBIENCE#GENERAL': 9, 
        'SERVICE#GENERAL': 10, 
        'LOCATION#GENERAL': 11 
    }
    X_Category = []
    Y_Category = []
    X_Polarity = []
    Y_Polarity = []
    for review in all_reviews:
        for sentence in review.sentences:
            for opinion in sentence.opinions_expected:
                feature_vec_for_sentences_opinion = []
                feature_vec_for_sentences_opinion_p = []
                for word in vocab:
                    if word in sentence.text:
                        feature_vec_for_sentences_opinion.append(1)
                        feature_vec_for_sentences_opinion_p.append(1)
                    else:
                        feature_vec_for_sentences_opinion.append(0)
                        feature_vec_for_sentences_opinion_p.append(0)
                X_Category.append(feature_vec_for_sentences_opinion)
                feature_vec_for_sentences_opinion_p.append(category_dict[opinion.category])
                X_Polarity.append(feature_vec_for_sentences_opinion_p)
                Y_Category.append(opinion.category)
                Y_Polarity.append(opinion.polarity)
    X_Category = np.array(X_Category)
    Y_Category = np.array(Y_Category)
    X_Polarity = np.array(X_Polarity)
    Y_Polarity = np.array(Y_Polarity)
    return X_Category, Y_Category, X_Polarity, Y_Polarity

def create_vocab(all_reviews, stop_words):
    vocab = {}
    all_vocab = {}
    all_pos = {}
    all_word_shapes = {}
    all_word_types = {}
    k = 1
    l = 1
    m = 1
    n = 1
    #print(len(all_reviews))
    for review in all_reviews:
        for sentence in review.sentences:
            words = sentence.text.split(' ')
            #print(words)
            for word in words:
                if word not in all_vocab:
                    all_vocab[word] = k
                    k += 1
                if word in stop_words:
                    continue
                if word in vocab:
                    vocab[word] += 1
                elif word not in vocab:
                    vocab[word] = 1
            for pos in sentence.pos_tags_on_words_in_sentence:
                if pos not in all_pos:
                    all_pos[pos] = l
                    l += 1
            for word_shape in sentence.word_shapes:
                if word_shape not in all_word_shapes:
                    all_word_shapes[word_shape] = m
                    m += 1
            for w_type in sentence.word_types:
                if w_type not in all_word_types:
                    all_word_types[w_type] = n
                    n += 1

    sorted_vocab = dict(sorted(vocab.items(), key=lambda item: item[1], reverse=True))
    end_vocab = []
    i = 0
    for key in sorted_vocab:
        if i >= 1000:
            break
        end_vocab.append(key)
        i += 1
    return end_vocab, all_vocab, all_pos, all_word_shapes, all_word_types




'''
===============================================================================================================
                                        START PROGRAM
===============================================================================================================
'''   

if __name__ == "__main__":
    #stop words from nltk
    stop_words = set(stopwords.words('english'))
    path_trial = 'trial_data/restaurants_trial_english_sl.xml'
    path_train = 'train_data/ABSA16_Restaurants_Train_SB1_v2.xml'
    path_test = 'test_gold_data/EN_REST_SB1_TEST.xml.gold'
    opinion_expected = True
    all_reviews = parsing.parse_xml(path_trial, opinion_expected, stop_words)
    process_reviews(all_reviews, stop_words)
    #all_test_reviews = parsing.parse_xml(path_test, opinion_expected, stop_words)
    #process_reviews_all(all_reviews, all_test_reviews, stop_words)
