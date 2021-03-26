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
from sklearn_crfsuite import CRF
from sklearn.model_selection import KFold
import re
import scoring
import output_file_creation
import parsing
import review_opinion_sent
import sys


nltk.download('stopwords')

    
def process_reviews(all_reviews, stop_words):
    end_vocab = create_vocab(all_reviews, stop_words)
    '''
    print(f'VOCAB: {end_vocab}')
    print()
    '''
    X_Category, Y_Category, X_Polarity, Y_Polarity = create_feature_vectors_and_expected_values(all_reviews, end_vocab)
    X_Target_BIO, Y_Target_BIO = create_feature_vectors_bio(all_reviews)
    #print(len(X_Target_BIO))
    #print(len(Y_Target_BIO))
    X_C_Train, X_C_Test, Y_C_Train, Y_C_Test = train_test_split(X_Category, Y_Category, random_state = 0, shuffle = False) #default 25% become test examples
    X_P_Train, X_P_Test, Y_P_train, Y_P_Test = train_test_split(X_Polarity, Y_Polarity, random_state = 0, shuffle = False)
    X_T_Train, X_T_Test, Y_T_Train, Y_T_Test = train_test_split(X_Target_BIO, Y_Target_BIO, random_state = 0, shuffle = False)
    #print(len(X_T_Train))
    #print(len(Y_T_Train))
    '''
    print(X_T_Train)
    print()
    print(Y_T_Train)
    print()
    '''
    svm_model_category = SVC(kernel='linear', C=1).fit(X_C_Train, Y_C_Train)
    svm_category_predictions = svm_model_category.predict(X_C_Test)
    accuracy = svm_model_category.score(X_C_Test, Y_C_Test)
    print(f'CATEGORY ACCURACY SVM: {accuracy}')
    svm_model_polarty = SVC(kernel='linear', C=1).fit(X_P_Train, Y_P_train)
    svm_polarity_predications = svm_model_polarty.predict(X_P_Test)
    accuracy = svm_model_polarty.score(X_P_Test, Y_P_Test)
    print(f'POLARITY ACCURACY SVM: {accuracy}')
    clf_category = OneVsRestClassifier(SVC(kernel='linear', C=1)).fit(X_C_Train, Y_C_Train)
    clf_category_predictions = clf_category.predict(X_C_Test)
    accuracy = clf_category.score(X_C_Test, Y_C_Test)
    print(f'CATEGORY ACCURACY 1vsRest SVM: {accuracy}')
    clf_polarity = OneVsRestClassifier(SVC(kernel='linear', C=1)).fit(X_P_Train, Y_P_train)
    clf_polarity_predictions = clf_polarity.predict(X_P_Test)
    accuracy = clf_polarity.score(X_P_Test, Y_P_Test)
    print(f'POLARITY ACCURACY 1vsResr SVM: {accuracy}')
    crf_model_target = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=False).fit(X_T_Train, Y_T_Train)
    crf_target_predictions = crf_model_target.predict(X_T_Test)
    accuracy = crf_model_target.score(X_T_Test, Y_T_Test)
    print(f'TARGET ACCURACY CRF: {accuracy}')
    print()

    predict_opinions(all_reviews, X_C_Train, Y_Target_BIO, svm_category_predictions, svm_polarity_predications, clf_category_predictions, clf_polarity_predictions, crf_target_predictions)
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
        '''
        print(train_index)
        for t_i, te_i in kf.split(X_Target_BIO):
            print(t_i)
            print(X_Target_BIO[0])
            print(X_Category[0])
            X_T_Train, X_T_Test = X_Target_BIO[t_i], X_Target_BIO[te_i]
            Y_T_Train, Y_T_Test = Y_Target_BIO[t_i], Y_Target_BIO[te_i]
        '''
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

        svm_model_category = SVC(kernel='linear', C=1).fit(X_C_Train, Y_C_Train)
        svm_category_predictions = svm_model_category.predict(X_C_Test)
        accuracy = svm_model_category.score(X_C_Test, Y_C_Test)
        print(f'CATEGORY ACCURACY SVM: {accuracy}')
        svm_model_polarty = SVC(kernel='linear', C=1).fit(X_P_Train, Y_P_train)
        svm_polarity_predications = svm_model_polarty.predict(X_P_Test)
        accuracy = svm_model_polarty.score(X_P_Test, Y_P_Test)
        print(f'POLARITY ACCURACY SVM: {accuracy}')
        clf_category = OneVsRestClassifier(SVC(kernel='linear', C=1)).fit(X_C_Train, Y_C_Train)
        clf_category_predictions = clf_category.predict(X_C_Test)
        accuracy = clf_category.score(X_C_Test, Y_C_Test)
        print(f'CATEGORY ACCURACY 1vsRest SVM: {accuracy}')
        clf_polarity = OneVsRestClassifier(SVC(kernel='linear', C=1)).fit(X_P_Train, Y_P_train)
        clf_polarity_predictions = clf_polarity.predict(X_P_Test)
        accuracy = clf_polarity.score(X_P_Test, Y_P_Test)
        print(f'POLARITY ACCURACY 1vsResr SVM: {accuracy}')
        crf_model_target = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=False).fit(X_T_Train, Y_T_Train)
        crf_target_predictions = crf_model_target.predict(X_T_Test)
        accuracy = crf_model_target.score(X_T_Test, Y_T_Test)
        print(f'TARGET ACCURACY CRF: {accuracy}')
        print()

        predict_opinions_cv(all_reviews, train_index, test_index, svm_category_predictions, svm_polarity_predications, clf_category_predictions, clf_polarity_predictions, crf_target_predictions)
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

def process_reviews_all(all_reviews, all_test_reviews, stop_words):
    end_vocab = create_vocab(all_reviews, stop_words)
    X_C_Train, Y_C_Train, X_P_Train, Y_P_train = create_feature_vectors_and_expected_values(all_reviews, end_vocab)
    X_T_Train, Y_T_Train = create_feature_vectors_bio(all_reviews)
    X_C_Test, Y_C_Test, X_P_Test, Y_P_Test = create_feature_vectors_and_expected_values(all_test_reviews, end_vocab)
    X_T_Test, Y_T_Test = create_feature_vectors_bio(all_test_reviews)

    svm_model_category = SVC(kernel='linear', C=1).fit(X_C_Train, Y_C_Train)
    svm_category_predictions = svm_model_category.predict(X_C_Test)
    accuracy = svm_model_category.score(X_C_Test, Y_C_Test)
    print(f'CATEGORY ACCURACY SVM: {accuracy}')
    svm_model_polarty = SVC(kernel='linear', C=1).fit(X_P_Train, Y_P_train)
    svm_polarity_predications = svm_model_polarty.predict(X_P_Test)
    accuracy = svm_model_polarty.score(X_P_Test, Y_P_Test)
    print(f'POLARITY ACCURACY SVM: {accuracy}')
    clf_category = OneVsRestClassifier(SVC(kernel='linear', C=1)).fit(X_C_Train, Y_C_Train)
    clf_category_predictions = clf_category.predict(X_C_Test)
    accuracy = clf_category.score(X_C_Test, Y_C_Test)
    print(f'CATEGORY ACCURACY 1vsRest SVM: {accuracy}')
    clf_polarity = OneVsRestClassifier(SVC(kernel='linear', C=1)).fit(X_P_Train, Y_P_train)
    clf_polarity_predictions = clf_polarity.predict(X_P_Test)
    accuracy = clf_polarity.score(X_P_Test, Y_P_Test)
    print(f'POLARITY ACCURACY 1vsResr SVM: {accuracy}')
    crf_model_target = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=False).fit(X_T_Train, Y_T_Train)
    crf_target_predictions = crf_model_target.predict(X_T_Test)
    accuracy = crf_model_target.score(X_T_Test, Y_T_Test)
    print(f'TARGET ACCURACY CRF: {accuracy}')
    print()

    predict_opinions_all(all_test_reviews, svm_category_predictions, svm_polarity_predications, clf_category_predictions, clf_polarity_predictions, crf_target_predictions)
    scoring.calculate_scores(all_test_reviews)



def predict_opinions_cv(all_reviews, train_index, test_index, svm_category_predictions, svm_polarity_predications, clf_category_predictions, clf_polarity_predictions, crf_target_predictions):
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
                category_pred = svm_category_predictions[predict_index]
                #category_pred = clf_category_predictions[predict_index]
                target = ''
                i += 1
                t_f = False
                for k in range(len(sentence.words)):
                    word = sentence.words[k]
                    if crf_target_predictions[predict_index][k] == 'B' and t_f == False and (k == 0 or crf_target_predictions[predict_index][k-1] == 'O'):
                        t_f = True
                        if k < len(sentence.words) - 1 and crf_target_predictions[predict_index][k+1] == 'I':
                            word += ' ' + sentence.words[k+1]
                            if k < len(sentence.words) - 2 and crf_target_predictions[predict_index][k+2] == 'I':
                                word += ' ' + sentence.words[k+2]
                                if k < len(sentence.words) - 3 and crf_target_predictions[predict_index][k+3] == 'I':
                                    word += ' ' + sentence.words[k+3]
                        target += word
                if target == '':
                    target = 'NULL'
                #classifier for each performs worse -- one vs rest is below this comment
                #opinion_predicted = review_opinion_sent.Opinion(sentence.sentence_id, sentence.review_id, target, '0', '0', clf_category_predictions[predict_index], clf_polarity_predictions[predict_index])
                opinion_predicted = review_opinion_sent.Opinion(sentence.sentence_id, sentence.review_id, target, '0', '0', svm_category_predictions[predict_index], svm_polarity_predications[predict_index])
                #opinion_predicted.print_attr()
                predict_index += 1
                sentence.opinions_predicted.append(opinion_predicted)

def predict_opinions_all(all_reviews, svm_category_predictions, svm_polarity_predications, clf_category_predictions, clf_polarity_predictions, crf_target_predictions):
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
                    if crf_target_predictions[predict_index][k] == 'B' and t_f == False and (k == 0 or crf_target_predictions[predict_index][k-1] == 'O'):
                        t_f = True
                        if k < len(sentence.words) - 1 and crf_target_predictions[predict_index][k+1] == 'I':
                            word += ' ' + sentence.words[k+1]
                            if k < len(sentence.words) - 2 and crf_target_predictions[predict_index][k+2] == 'I':
                                word += ' ' + sentence.words[k+2]
                                if k < len(sentence.words) - 3 and crf_target_predictions[predict_index][k+3] == 'I':
                                    word += ' ' + sentence.words[k+3]
                        target += word
                if target == '':
                    target = 'NULL'
                #classifier for each performs worse -- one vs rest is below this comment
                #opinion_predicted = review_opinion_sent.Opinion(sentence.sentence_id, sentence.review_id, target, '0', '0', clf_category_predictions[predict_index], clf_polarity_predictions[predict_index])
                opinion_predicted = review_opinion_sent.Opinion(sentence.sentence_id, sentence.review_id, target, '0', '0', svm_category_predictions[predict_index], svm_polarity_predications[predict_index])
                #opinion_predicted.print_attr()
                predict_index += 1
                sentence.opinions_predicted.append(opinion_predicted)
def predict_opinions(all_reviews, X_C_Train, Y_Target_BIO, svm_category_predictions, svm_polarity_predications, clf_category_predictions, clf_polarity_predictions, crf_target_predictions):    
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
                category_pred = svm_category_predictions[predict_index]
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
                    if crf_target_predictions[predict_index][k] == 'B' and t_f == False and (k == 0 or crf_target_predictions[predict_index][k-1] == 'O'):
                        t_f = True
                        if k < len(sentence.words) - 1 and crf_target_predictions[predict_index][k+1] == 'I':
                            word += ' ' + sentence.words[k+1]
                            if k < len(sentence.words) - 2 and crf_target_predictions[predict_index][k+2] == 'I':
                                word += ' ' + sentence.words[k+2]
                                if k < len(sentence.words) - 3 and crf_target_predictions[predict_index][k+3] == 'I':
                                    word += ' ' + sentence.words[k+3]
                        target += word
                if target == '':
                    target = 'NULL'
                #classifier for each performs worse -- one vs rest is below this comment
                #opinion_predicted = review_opinion_sent.Opinion(sentence.sentence_id, sentence.review_id, target, '0', '0', clf_category_predictions[predict_index], clf_polarity_predictions[predict_index])
                opinion_predicted = review_opinion_sent.Opinion(sentence.sentence_id, sentence.review_id, target, '0', '0', svm_category_predictions[predict_index], svm_polarity_predications[predict_index])
                #opinion_predicted.print_attr()
                predict_index += 1
                sentence.opinions_predicted.append(opinion_predicted)

#Could expand BIO to match on the specific categories as well
def create_feature_vectors_bio(all_reviews):
    X_BIO = []
    Y_BIO = []
    for review in all_reviews:
        for sentence in review.sentences:
            for opinion in sentence.opinions_expected:
                answer = []
                word_feature_vec = []
                for i in range(len(sentence.words)):
                    prev_word = ''
                    prev_word_2 = ''
                    prev_pos = ''
                    prev_pos_2 = ''
                    prev_shape = ''
                    prev_shape_2 = ''
                    prev_type = ''
                    prev_type_2 = ''
                    if i > 0:
                        prev_word = sentence.words[i-1]
                        prev_pos = sentence.pos_tags_on_words_in_sentence[i-1][1]
                        prev_shape = sentence.word_shapes[i-1]
                        prev_type = sentence.word_types[i-1]
                    else:
                        prev_word = "SIGMA"
                        prev_pos = 'SIGMA_POS'
                        prev_shape = 'SIGMA_SHAPE'
                        prev_type = 'SIGMA_TYPE'
                    if i > 1:
                        prev_word_2 = sentence.words[i-2]
                        prev_pos_2 = sentence.pos_tags_on_words_in_sentence[i-2][1]
                        prev_shape_2 = sentence.word_shapes[i-2]
                        prev_type_2 = sentence.word_types[i-2]
                    else:
                        prev_word_2 = 'SIGMA'
                        prev_pos_2 = 'SIGMA_POS'
                        prev_shape_2 = 'SIGMA_SHAPE'
                        prev_type_2 = 'SIGMA_TYPE'
                    next_word = ''
                    next_word_2 = ''
                    next_word_3 = ''
                    next_pos = ''
                    next_pos_2 = ''
                    next_shape = ''
                    next_shape_2 = ''
                    next_type = ''
                    next_type_2 = ''
                    if i < len(sentence.words) - 1:
                        next_word = sentence.words[i+1]
                        next_pos = sentence.pos_tags_on_words_in_sentence[i+1][1]
                        next_shape = sentence.word_shapes[i+1]
                        next_type = sentence.word_types[i+1]
                    else:
                        next_word = "OMEGA"
                        next_pos = 'OMEGA_POS'
                        next_shape = 'OMEGA_SHAPE'
                        next_type = 'OMEGA_TYPE'
                    if i < len(sentence.words) - 2:
                        next_word_2 = sentence.words[i+2]
                        next_pos_2 = sentence.pos_tags_on_words_in_sentence[i+2][1]
                        next_shape_2 = sentence.word_shapes[i+2]
                        next_type_2 = sentence.word_types[i+2]
                    else:
                        next_word_2 = 'OMEGA'
                        next_pos_2 = 'OMEGA_POS'
                        next_shape_2 = 'OMEGA_SHAPE'
                        next_type_2 = 'OMEGA_TYPE'
                    if i < len(sentence.words) - 3:
                        next_word_3 = sentence.words[i+3]
                    else:
                        next_word_3 = 'OMEGA'
                    word = sentence.words[i]
                    pos = sentence.pos_tags_on_words_in_sentence[i][1] #[i] is tuple ie ('Would', 'MD')
                    word_shape = sentence.word_shapes[i]
                    word_type = sentence.word_types[i]
                    is_stop_word = sentence.words_are_stop_words[i]
                    feature_vec = {
                        'bias': 1.0,
                        'word': word, 
                        'pos': pos, 
                        'word_shape': word_shape, 
                        'word_type': word_type, 
                        'is_stop_word': is_stop_word,
                        #suffix
                        # prefix 
                        'prev_word': prev_word, 
                        'prev_word_2': prev_word_2, 
                        'next_word': next_word, 
                        'next_word_2': next_word_2, 
                        'next_word_3': next_word_3,
                        'prev_pos': prev_pos,
                        'prev_pos_2': prev_pos_2,
                        'prev_shape': prev_shape,
                        'prev_shape_2': prev_shape_2,
                        'prev_type': prev_type,
                        'prev_type_2': prev_type_2,
                        'next_pos': next_pos,
                        'next_pos_2': next_pos_2,
                        'next_shape': next_shape,
                        'next_shape_2': next_shape_2,
                        'next_type': next_type,
                        'next_type_2': next_type_2
                    }
                    target_words = opinion.target.split()
                    target_found = False
                    for j in range(len(target_words)):
                        if target_words[j] == word and j == 0:
                            target_found = True
                            answer.append('B')
                        elif target_words[j] == word and j >= 1 and target_found == False:
                            target_found = True
                            answer.append('I')
                    if target_found == False:
                        answer.append('O')
                    word_feature_vec.append(feature_vec)
                X_BIO.append(word_feature_vec)
                Y_BIO.append(answer)
    #X_BIO = np.array(X_BIO)
    #Y_BIO = np.array(Y_BIO)
    return X_BIO, Y_BIO
                        

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
    #print(len(all_reviews))
    for review in all_reviews:
        for sentence in review.sentences:
            words = sentence.text.split(' ')
            #print(words)
            for word in words:
                if word in stop_words:
                    continue
                if word in vocab:
                    vocab[word] += 1
                elif word not in vocab:
                    vocab[word] = 1
    sorted_vocab = dict(sorted(vocab.items(), key=lambda item: item[1], reverse=True))
    end_vocab = []
    i = 0
    for key in sorted_vocab:
        if i >= 1000:
            break
        end_vocab.append(key)
        i += 1
    return end_vocab




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
    all_reviews = parsing.parse_xml(path_train, opinion_expected, stop_words)
    #process_reviews(all_reviews, stop_words)
    all_test_reviews = parsing.parse_xml(path_test, opinion_expected, stop_words)
    process_reviews_all(all_reviews, all_test_reviews, stop_words)
