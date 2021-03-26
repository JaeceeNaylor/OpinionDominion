import xml.etree.ElementTree as ET 
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
from sklearn.model_selection import KFold
import scoring
import output_file_creation
import parsing
import review_opinion_sent


nltk.download('stopwords')

    
def process_reviews(all_reviews, stop_words):
    end_vocab = create_vocab(all_reviews, stop_words)
    '''
    print(f'VOCAB: {end_vocab}')
    print()
    '''
    X_Category, Y_Category, X_Polarity, Y_Polarity = create_feature_vectors_and_expected_values(all_reviews, end_vocab)
    X_C_Train, X_C_Test, Y_C_Train, Y_C_Test = train_test_split(X_Category, Y_Category, random_state = 0, shuffle = False) #default 25% become test examples
    X_P_Train, X_P_Test, Y_P_train, Y_P_Test = train_test_split(X_Polarity, Y_Polarity, random_state = 0, shuffle = False)
    svm_model_category = SVC(kernel='linear', C=1).fit(X_C_Train, Y_C_Train)
    svm_category_predictions = svm_model_category.predict(X_C_Test)
    accuracy = svm_model_category.score(X_C_Test, Y_C_Test)
    print(f'CATEGORY ACCURACY: {accuracy}')
    svm_model_polarty = SVC(kernel='linear', C=1).fit(X_P_Train, Y_P_train)
    svm_polarity_predications = svm_model_polarty.predict(X_P_Test)
    accuracy = svm_model_polarty.score(X_P_Test, Y_P_Test)
    print(f'POLARITY ACCURACY: {accuracy}')
    clf_category = OneVsRestClassifier(SVC(kernel='linear', C=1)).fit(X_C_Train, Y_C_Train)
    clf_category_predictions = clf_category.predict(X_C_Test)
    accuracy = clf_category.score(X_C_Test, Y_C_Test)
    print(f'CATEGORY ACCURACY: {accuracy}')
    clf_polarity = OneVsRestClassifier(SVC(kernel='linear', C=1)).fit(X_P_Train, Y_P_train)
    clf_polarity_predictions = clf_polarity.predict(X_P_Test)
    accuracy = clf_polarity.score(X_P_Test, Y_P_Test)
    print(f'POLARITY ACCURACY: {accuracy}')
    print()

    predict_opinions(all_reviews, X_C_Train, svm_category_predictions, svm_polarity_predications, clf_category_predictions, clf_polarity_predictions)
    scoring.calculate_scores(all_reviews)

    #cross validation
    #cross validation
    kf = KFold(n_splits = 5)
    kf.get_n_splits(X_Category)
    kf.get_n_splits(X_Polarity)
    fold = 1
    for train_index, test_index in kf.split(X_Category):
        print()
        print(f'FOLD: {fold}')
        fold += 1
        X_C_Train, X_C_Test = X_Category[train_index], X_Category[test_index]
        Y_C_Train, Y_C_Test = Y_Category[train_index], Y_Category[test_index]

        X_P_Train, X_P_Test = X_Polarity[train_index], X_Polarity[test_index]
        Y_P_train, Y_P_Test = Y_Polarity[train_index], Y_Polarity[test_index]

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
        print()

        predict_opinions_cv(all_reviews, train_index, test_index, svm_category_predictions, svm_polarity_predications, clf_category_predictions, clf_polarity_predictions)
        scoring.calculate_scores(all_reviews)

    target_predicted_file_name = 'output_target_data/trial_SVM.target.predicted'
    target_expected_file_name = 'output_target_data/trial_SVM.target.expected'
    output_file_creation.create_files(target_predicted_file_name, target_expected_file_name, all_reviews, 'TARGET')

    polarity_predicted_file_name = 'output_polarity_data/trial_SVM.target.predicted'
    polarity_expected_file_name = 'output_polarity_data/trial_SVM.target.expected'
    output_file_creation.create_files(polarity_predicted_file_name, polarity_expected_file_name, all_reviews, 'POLARITY')

    category_predicted_file_name = 'output_category_data/trial_SVM.target.predicted'
    category_expected_file_name = 'output_category_data/trial_SVM.target.expected'
    output_file_creation.create_files(category_predicted_file_name, category_expected_file_name, all_reviews, 'CATEGORY')

def predict_opinions_cv(all_reviews, train_index, test_index, svm_category_predictions, svm_polarity_predications, clf_category_predictions, clf_polarity_predictions):
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
                else:
                    i += 1
    i = 0
    for review in all_reviews:
        for sentence in review.sentences:
            sentence.opinions_predicted = []
            for opinion in sentence.opinions_expected:
                if i in test_i_d:
                    i += 1
                    category_pred = svm_category_predictions[predict_index]
                    #category_pred = clf_category_predictions[predict_index]
                    target = 'NULL'
                    for word in sentence.text.split(' '):
                        if category_pred in category_to_ote:
                            if word in category_to_ote[category_pred]:
                                target = word
                        else:
                            continue
                    #classifier for each performs worse
                    #opinion_predicted = Opinion(sentence.sentence_id, sentence.review_id, target, '0', '0', clf_category_predictions[predict_index], clf_polarity_predictions[predict_index])
                    opinion_predicted = review_opinion_sent.Opinion(sentence.sentence_id, sentence.review_id, target, '0', '0', svm_category_predictions[predict_index], svm_polarity_predications[predict_index])
                    #opinion_predicted.print_attr()
                    predict_index += 1
                    sentence.opinions_predicted.append(opinion_predicted)
                else:
                    i += 1

def predict_opinions(all_reviews, X_C_Train, svm_category_predictions, svm_polarity_predications, clf_category_predictions, clf_polarity_predictions):    
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
                i += 1
                category_pred = svm_category_predictions[predict_index]
                #category_pred = clf_category_predictions[predict_index]
                target = 'NULL'
                for word in sentence.text.split(' '):
                    if word in category_to_ote[category_pred]:
                        target = word
                #classifier for each performs worse
                #opinion_predicted = Opinion(sentence.sentence_id, sentence.review_id, target, '0', '0', clf_category_predictions[predict_index], clf_polarity_predictions[predict_index])
                opinion_predicted = review_opinion_sent.Opinion(sentence.sentence_id, sentence.review_id, target, '0', '0', svm_category_predictions[predict_index], svm_polarity_predications[predict_index])
                #opinion_predicted.print_attr()
                predict_index += 1
                sentence.opinions_predicted.append(opinion_predicted)

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
    stop_words = set(stopwords.words('english'))
    path_trial = 'trial_data/restaurants_trial_english_sl.xml'
    path_train = 'train_data/ABSA16_Restaurants_Train_SB1_v2.xml'
    path_test = 'test_gold_data/EN_REST_SB1_TEST.xml.gold'
    opinion_expected = True
    all_reviews = parsing.parse_xml(path_train, opinion_expected, stop_words)
    process_reviews(all_reviews, stop_words)
