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
from sklearn_crfsuite import CRF
import re


nltk.download('stopwords')


'''
===============================================================================================================
                                                CLASSES
===============================================================================================================
'''   

class Review:
    def __init__(self, review_id, sentences):
        self.review_id = review_id
        self.sentences = sentences

    def print_attr(self):
        print(f'REVIEW_ID: {self.review_id}')
        print('SENTENCES:')
        for sentence in self.sentences:
            sentence.print_attr()
        print()

class Sentence:
    def __init__(self, sentence_id, review_id, text, opinions_predicted, opinions_expected, stop_words):
        self.sentence_id = sentence_id
        self.review_id = review_id
        self.text = text
        self.lowercase_text = text.lower()
        self.words = text.split()
        self.lowercase_words = self.lowercase_text.split()
        '''
        #not sure if following two lines are necessary because resaurant data 
        #should have sentences separated already
        self.sentence_tokens = nltk.sent_tokenize(text)
        self.sentence_tokens_lowercase = nltk.sent_tokenize(self.lowercase_text)
        if len(self.sentence_tokens) > 1:
            print(f'ERROR: More than one sentence in sentence for SENTENCE_ID: {self.sentence_id}')
        '''
        self.tokenized_words_in_sentence = nltk.word_tokenize(self.text)
        self.tokenized_words_in_sentence_lowercase = nltk.word_tokenize(self.lowercase_text)
        self.pos_tags_on_words_in_sentence = nltk.pos_tag(self.tokenized_words_in_sentence)
        self.pos_tags_on_words_in_sentence_lowercase = nltk.pos_tag(self.tokenized_words_in_sentence_lowercase)
        self.opinions_predicted = opinions_predicted
        self.opinions_expected = opinions_expected
        self.word_shapes = []
        self.words_are_stop_words = []
        self.word_types = []
        string_check = re.compile('[@_!#$%^&*()<>?/\|}{~:]')
        for word in self.tokenized_words_in_sentence:
            self.word_shapes.append(self.word_shape(word))
            if word in stop_words:
                self.words_are_stop_words.append(1)
            else:
                self.words_are_stop_words.append(0)
            if string_check.search(word) != None:
                self.word_types.append('S')
            elif word[0].isupper():
                self.word_types.append('U')
            elif word.isdigit():
                self.word_types.append('D')
            elif word.isalnum() and word.isalpha() == False:
                self.word_types.append('C')
            else:
                self.word_types.append('N')
            
        
    
    def word_shape(self, text):
        t1 = re.sub('[A-Z]', 'X', text)
        t2 = re.sub('[a-z]', 'x', t1)
        t3 = re.sub('[.!?,;:]', 'p', t2)
        return re.sub('[0-9]', 'd', t3)
    
    def print_attr(self):
        print(f'REVIEW_ID: {self.review_id}')
        print(f'SENTENCE_ID: {self.sentence_id}')
        print(f'TEXT: {self.text}')
        print(f'TEXT_LOWER: {self.lowercase_text}')
        print(f'WORDS: {self.words}')
        print(f'WORDS_LOWER: {self.lowercase_words}')
        print(f'TOKEN WORDS: {self.tokenized_words_in_sentence}')
        print(f'POS WORDS: {self.pos_tags_on_words_in_sentence}')
        print(f'WORD SHAPES: {self.word_shapes}')
        print(f'WORD STOP: {self.words_are_stop_words}')
        print(f'WORD TYPES: {self.word_types}')
        print('OPINIONS PREDICTED')
        for opinion_pred in self.opinions_predicted:
            opinion_pred.print_attr()
            print()
        print('OPINIONS EXPECTED')
        for opinion_exp in self.opinions_expected:
            opinion_exp.print_attr()
            print()
            

class Opinion:
    def __init__(self, sentence_id, review_id, target, from_index, to_index, category, polarity):
        self.sentence_id = sentence_id
        self.review_id = review_id
        self.target = target
        self.target_begin_index = from_index
        self.target_end_index = to_index
        self.category = category
        self.polarity = polarity
        self.correctly_labeled_target_words = 0
        self.total_target_words = len(target.split())
    
    def print_attr(self):
        print(f'REVIEW_ID: {self.review_id}')
        print(f'SENTENCE_ID: {self.sentence_id}')
        print(f'TARGET OTE: {self.target}')
        print(f'TARGET START INDEX: {self.target_begin_index}')
        print(f'TARGET END INDEX: {self.target_end_index}')
        print(f'CATEGORY: {self.category}')
        print(f'POLARITY: {self.polarity}')

    def string_target_attr(self):
        return f'SentenceID: {self.sentence_id}\nTarget: {self.target}\nTargetStartIndex: {self.target_begin_index}\nTargetEndIndex: {self.target_end_index}\n\n'

    def string_polarity_attr(self):
        return f'SentenceID: {self.sentence_id}\nPolarity: {self.polarity}\n\n'
    
    def string_category_attr(self):
        return f'SentenceID: {self.sentence_id}\nCategory: {self.category}\n\n'








'''
===============================================================================================================
                                    PARSE XML and CREATE CLASS INSTANCES
===============================================================================================================
'''   

def parse_xml(xml_file, had_opinion_expected, stop_words):
    #begin parsing XML
    tree = ET.parse(xml_file)
    reviews = tree.getroot()
    all_reviews = []
    for review in reviews:
        review_id = review.attrib['rid']
        sentences = review[0]
        sentences_for_review_object = []
        for sentence in sentences:
            if 'OutOfScope' in sentence.attrib:
                continue
            sentence_id = sentence.attrib['id']
            text = sentence[0].text
            opinions_expected = []
            opinions_predicted = []
            if len(sentence) > 1:
                had_opinion_expected = True
                #extract the golden opinion criteria
                opinions = sentence[1]
                for stated_opinion in opinions:
                    #logic to find expected opinion results
                    target, target_begin_index, target_end_index  = stated_opinion.attrib['target'], stated_opinion.attrib['from'], stated_opinion.attrib['to']
                    category, polarity = stated_opinion.attrib['category'], stated_opinion.attrib['polarity']
                    opinion_expected = Opinion(sentence_id, review_id, target, target_begin_index, target_end_index, category, polarity)
                    opinions_expected.append(opinion_expected)
            else:
                #this sentence could not have an opinion or this could not be a golden file
                if had_opinion_expected:
                    continue
                #no opinion that is golden to extract
                else:
                    #begin logic to parse our own opinion results
                    #opinion_predicted = predict_opinion(sentence_id, review_id, text)
                    #opinions_predicted.append(opinion_predicted)
                    print('PROBLEM')
            sentence_gathered = Sentence(sentence_id, review_id, text, opinions_predicted, opinions_expected, stop_words)
            #sentence_gathered.print_attr()
            sentences_for_review_object.append(sentence_gathered)
        review_collected = Review(review_id, sentences_for_review_object)
        all_reviews.append(review_collected)
    return all_reviews
    
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
    print()
    svm_model_polarty = SVC(kernel='linear', C=1).fit(X_P_Train, Y_P_train)
    svm_polarity_predications = svm_model_polarty.predict(X_P_Test)
    accuracy = svm_model_polarty.score(X_P_Test, Y_P_Test)
    print(f'POLARITY ACCURACY SVM: {accuracy}')
    print()
    clf_category = OneVsRestClassifier(SVC(kernel='linear', C=1)).fit(X_C_Train, Y_C_Train)
    clf_category_predictions = clf_category.predict(X_C_Test)
    accuracy = clf_category.score(X_C_Test, Y_C_Test)
    print(f'CATEGORY ACCURACY 1vsRest SVM: {accuracy}')
    print()
    clf_polarity = OneVsRestClassifier(SVC(kernel='linear', C=1)).fit(X_P_Train, Y_P_train)
    clf_polarity_predictions = clf_polarity.predict(X_P_Test)
    accuracy = clf_polarity.score(X_P_Test, Y_P_Test)
    print(f'POLARITY ACCURACY 1vsResr SVM: {accuracy}')
    print()
    crf_model_target = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=False).fit(X_T_Train, Y_T_Train)
    crf_target_predictions = crf_model_target.predict(X_T_Test)
    accuracy = crf_model_target.score(X_T_Test, Y_T_Test)
    print(f'TARGET ACCURACY CRF: {accuracy}')
    print()

    predict_opinions(all_reviews, X_C_Train, Y_Target_BIO, svm_category_predictions, svm_polarity_predications, clf_category_predictions, clf_polarity_predictions, crf_target_predictions)
    
    
    calculate_scores(all_reviews)
    target_predicted_file_name = 'output_target_data/trial_CRF.target.predicted'
    target_expected_file_name = 'output_target_data/trial_CRF.target.expected'
    create_files(target_predicted_file_name, target_expected_file_name, all_reviews, 'TARGET')

    polarity_predicted_file_name = 'output_polarity_data/trial_CRF.target.predicted'
    polarity_expected_file_name = 'output_polarity_data/trial_CRF.target.expected'
    create_files(polarity_predicted_file_name, polarity_expected_file_name, all_reviews, 'POLARITY')

    category_predicted_file_name = 'output_category_data/trial_CRF.target.predicted'
    category_expected_file_name = 'output_category_data/trial_CRF.target.expected'
    create_files(category_predicted_file_name, category_expected_file_name, all_reviews, 'CATEGORY')


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
                #classifier for each performs worse
                #opinion_predicted = Opinion(sentence.sentence_id, sentence.review_id, target, '0', '0', clf_category_predictions[predict_index], clf_polarity_predictions[predict_index])
                opinion_predicted = Opinion(sentence.sentence_id, sentence.review_id, target, '0', '0', svm_category_predictions[predict_index], svm_polarity_predications[predict_index])
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
                                            Create Output Files
===============================================================================================================
'''   

def create_files(predicted_file_name, expected_file_name, all_reviews, type_file):
    predicted_file = open(predicted_file_name, "w")
    expected_file = open(expected_file_name, "w")
    for review in all_reviews:
        if type_file == 'TARGET':
            for sentence in review.sentences:
                if len(sentence.opinions_predicted) > 0:
                    for opinion_pred in sentence.opinions_predicted:
                        predicted_file.write(opinion_pred.string_target_attr())
                    for opinion_exp in sentence.opinions_expected:
                        expected_file.write(opinion_exp.string_target_attr())
        elif type_file == 'POLARITY':
            for sentence in review.sentences:
                if len(sentence.opinions_predicted) > 0:
                    for opinion_pred in sentence.opinions_predicted:
                        predicted_file.write(opinion_pred.string_polarity_attr())
                    for opinion_exp in sentence.opinions_expected:
                        expected_file.write(opinion_exp.string_polarity_attr())
        elif type_file == 'CATEGORY':
            for sentence in review.sentences:
                if len(sentence.opinions_predicted) > 0:
                    for opinion_pred in sentence.opinions_predicted:
                        predicted_file.write(opinion_pred.string_category_attr())
                    for opinion_exp in sentence.opinions_expected:
                        expected_file.write(opinion_exp.string_category_attr())
    predicted_file.close()
    expected_file.close()










'''
===============================================================================================================
                                            Calculate Feature Scores
===============================================================================================================
'''   

def calculate_scores(all_reviews):
    target_recall, target_precision = calcuate_target_recall_precision(all_reviews)
    target_fmeasure = fmeasure_score(target_recall, target_precision)
    print(f'TARGET RECALL: {target_recall}')
    print(f'TARGET PERECISION: {target_precision}')
    print(f'TARGET F-MEASURE: {target_fmeasure}')
    print()
    entity_recall, entity_precision = calcuate_entity_recall_precision(all_reviews)
    entity_fmeasure = fmeasure_score(entity_recall, entity_precision)
    print(f'ENTITY RECALL: {entity_recall}')
    print(f'ENTITY PERECISION: {entity_precision}')
    print(f'ENTITY F-MEASURE: {entity_fmeasure}')
    print()
    attribute_recall, attribute_precision = calcuate_attribute_recall_precision(all_reviews)
    attribute_fmeasure = fmeasure_score(attribute_recall, attribute_precision)
    print(f'ATTRIBUTE RECALL: {attribute_recall}')
    print(f'ATTRIBUTE PERECISION: {attribute_precision}')
    print(f'ATTRIBUTE F-MEASURE: {attribute_fmeasure}')
    print()
    e_a_recall, e_a_precision = calcuate_e_a_recall_precision(all_reviews)
    e_a_fmeasure = fmeasure_score(e_a_recall, e_a_precision)
    print(f'E#A (CATEGORY) RECALL: {e_a_recall}')
    print(f'E#A (CATEGORY) PERECISION: {e_a_precision}')
    print(f'E#A (CATEGORY) F-MEASURE: {e_a_fmeasure}')
    print()
    polarity_recall, polarity_precision = calcuate_polarity_recall_precision(all_reviews)
    polarity_fmeasure = fmeasure_score(polarity_recall, polarity_precision)
    print(f'POLARITY RECALL: {polarity_recall}')
    print(f'POLARITY PERECISION: {polarity_precision}')
    print(f'POLARITY F-MEASURE: {polarity_fmeasure}')
    print()

#recall is the number of correct words generated by IE system / the total number of words in the answer string.
#precision is the number of correct words generated by IE system / the total number of words generated by IE system
def calcuate_target_recall_precision(all_reviews):
    total_words_opinion_target_pred = 0
    total_words_opinion_target_exp = 0
    total_correctly_labeled = 0
    for review in all_reviews:
        for sentence in review.sentences:
            for i in range(0, len(sentence.opinions_predicted)):
                opinion_pred = sentence.opinions_predicted[i]
                opinion_exp = sentence.opinions_expected[i]
                total_words_opinion_target_pred += opinion_pred.total_target_words
                total_words_opinion_target_exp += opinion_exp.total_target_words
                total_correctly_labeled += find_correctly_labeled_words_target(opinion_pred.target, opinion_exp.target)
    recall = total_correctly_labeled / total_words_opinion_target_exp
    percision = total_correctly_labeled / total_words_opinion_target_pred
    return recall, percision

def find_correctly_labeled_words_target(target_pred, target_exp):
    target_pred_words = target_pred.split()
    target_exp_words = target_exp.split()
    correctly_predicted = 0
    for word in target_pred_words:
        if word in target_exp_words:
            correctly_predicted += 1
    return correctly_predicted

#(2*percison*recall)/(recall+percision) this is the harmonic mean of reacll and percision
def fmeasure_score(target_recall, target_precision):
    if (target_recall+target_precision) == 0:
        return 0.0
    fmeasure = (2*target_precision*target_recall)/(target_recall+target_precision)
    return fmeasure

'''precision is the fraction of events where we correctly declared 𝑖 / instances where the algorithm declared 𝑖.
recall is the fraction of events where we correctly declared 𝑖 / of the cases where the true of state of the world is 𝑖.
'''
def calcuate_entity_recall_precision(all_reviews):
    entities_possible = {'FOOD', 'DRINKS', 'SERVICE', 'AMBIENCE', 'LOCATION', 'RESTAURANT'}
    recalls_added_together = 0
    precisions_added_together = 0
    algorithm_amounts_of_entity_type = {}
    expected_amounts_of_entity_type = {}
    correct_amounts_of_entity_type = {}
    for review in all_reviews:
        for sentence in review.sentences:
            for i in range(0, len(sentence.opinions_predicted)):
                opinion_pred = sentence.opinions_predicted[i]
                opinion_exp = sentence.opinions_expected[i]
                entity_pred = opinion_pred.category.split('#')[0]
                entity_exp = opinion_exp.category.split('#')[0]
                if entity_exp in expected_amounts_of_entity_type:
                    expected_amounts_of_entity_type[entity_exp] += 1
                else:
                    expected_amounts_of_entity_type[entity_exp] = 1
                if entity_pred in algorithm_amounts_of_entity_type:
                    algorithm_amounts_of_entity_type[entity_pred] += 1
                else:
                    algorithm_amounts_of_entity_type[entity_pred] = 1
                if entity_exp == entity_pred:
                    if entity_pred in correct_amounts_of_entity_type:
                        correct_amounts_of_entity_type[entity_pred] += 1
                    else:
                        correct_amounts_of_entity_type[entity_pred] = 1
    total_entity_labels = len(correct_amounts_of_entity_type)
    for entity_type in entities_possible:
        if entity_type in correct_amounts_of_entity_type:
            recall = correct_amounts_of_entity_type[entity_type] / expected_amounts_of_entity_type[entity_type]
            recalls_added_together += recall
            percision = correct_amounts_of_entity_type[entity_type] / algorithm_amounts_of_entity_type[entity_type]
            precisions_added_together += percision
    if total_entity_labels == 0:
        return 0.0, 0.0
    average_percision = precisions_added_together / total_entity_labels
    average_recall = recalls_added_together / total_entity_labels
    return average_recall, average_percision

'''precision is the fraction of events where we correctly declared 𝑖 / instances where the algorithm declared 𝑖.
recall is the fraction of events where we correctly declared 𝑖 / of the cases where the true of state of the world is 𝑖.
'''
def calcuate_attribute_recall_precision(all_reviews):
    attributes_possible = {'GENERAL', 'PRICES', 'QUALITY', 'STYLE_OPTIONS', 'MISCELLANEOUS'}
    recalls_added_together = 0
    precisions_added_together = 0
    algorithm_amounts_of_attribute_type = {}
    expected_amounts_of_attribute_type = {}
    correct_amounts_of_attribute_type = {}
    for review in all_reviews:
        for sentence in review.sentences:
            for i in range(0, len(sentence.opinions_predicted)):
                opinion_pred = sentence.opinions_predicted[i]
                opinion_exp = sentence.opinions_expected[i]
                attr_pred = opinion_pred.category.split('#')[1]
                attr_exp = opinion_exp.category.split('#')[1]
                if attr_exp in expected_amounts_of_attribute_type:
                    expected_amounts_of_attribute_type[attr_exp] += 1
                else:
                    expected_amounts_of_attribute_type[attr_exp] = 1
                if attr_pred in algorithm_amounts_of_attribute_type:
                    algorithm_amounts_of_attribute_type[attr_pred] += 1
                else:
                    algorithm_amounts_of_attribute_type[attr_pred] = 1
                if attr_exp == attr_pred:
                    if attr_pred in correct_amounts_of_attribute_type:
                        correct_amounts_of_attribute_type[attr_pred] += 1
                    else:
                        correct_amounts_of_attribute_type[attr_pred] = 1
    total_attr_labels = len(correct_amounts_of_attribute_type)
    for attr_type in attributes_possible:
        if attr_type in correct_amounts_of_attribute_type:
            recall = correct_amounts_of_attribute_type[attr_type] / expected_amounts_of_attribute_type[attr_type]
            recalls_added_together += recall
            percision = correct_amounts_of_attribute_type[attr_type] / algorithm_amounts_of_attribute_type[attr_type]
            precisions_added_together += percision
    if total_attr_labels == 0:
        return 0.0, 0.0
    average_percision = precisions_added_together / total_attr_labels
    average_recall = recalls_added_together / total_attr_labels
    return average_recall, average_percision

'''precision is the fraction of events where we correctly declared 𝑖 / instances where the algorithm declared 𝑖.
recall is the fraction of events where we correctly declared 𝑖 / of the cases where the true of state of the world is 𝑖.
'''
def calcuate_e_a_recall_precision(all_reviews):
    attributes_possible = {
        'RESTAURANT#GENERAL', 'RESTAURANT#PRICES', 'RESTAURANT#MISCELLANEOUS', 
        'FOOD#PRICES', 'FOOD#QUALITY', 'FOOD#STYLE_OPTIONS', 
        'DRINKS#PRICES', 'DRINKS#QUALITY', 'DRINKS#STYLE_OPTIONS',    
        'AMBIENCE#GENERAL', 
        'SERVICE#GENERAL', 
        'LOCATION#GENERAL' 
    }
    recalls_added_together = 0
    precisions_added_together = 0
    algorithm_amounts_of_attribute_type = {}
    expected_amounts_of_attribute_type = {}
    correct_amounts_of_attribute_type = {}
    for review in all_reviews:
        for sentence in review.sentences:
            for i in range(0, len(sentence.opinions_predicted)):
                opinion_pred = sentence.opinions_predicted[i]
                opinion_exp = sentence.opinions_expected[i]
                attr_pred = opinion_pred.category
                attr_exp = opinion_exp.category
                if attr_exp in expected_amounts_of_attribute_type:
                    expected_amounts_of_attribute_type[attr_exp] += 1
                else:
                    expected_amounts_of_attribute_type[attr_exp] = 1
                if attr_pred in algorithm_amounts_of_attribute_type:
                    algorithm_amounts_of_attribute_type[attr_pred] += 1
                else:
                    algorithm_amounts_of_attribute_type[attr_pred] = 1
                if attr_exp == attr_pred:
                    if attr_pred in correct_amounts_of_attribute_type:
                        correct_amounts_of_attribute_type[attr_pred] += 1
                    else:
                        correct_amounts_of_attribute_type[attr_pred] = 1
    total_attr_labels = len(correct_amounts_of_attribute_type)
    for attr_type in attributes_possible:
        if attr_type in correct_amounts_of_attribute_type:
            '''
            print(f'E#A TYPE: {attr_type}')
            print(f'CORRECT GUESSES: {correct_amounts_of_attribute_type}')
            print(f'EXPECTED: {expected_amounts_of_attribute_type}')
            print(f'PREDICTED: {algorithm_amounts_of_attribute_type}')
            print()
            '''
            recall = correct_amounts_of_attribute_type[attr_type] / expected_amounts_of_attribute_type[attr_type]
            recalls_added_together += recall
            percision = correct_amounts_of_attribute_type[attr_type] / algorithm_amounts_of_attribute_type[attr_type]
            precisions_added_together += percision
    if total_attr_labels == 0:
        return 0.0, 0.0
    average_percision = precisions_added_together / total_attr_labels
    average_recall = recalls_added_together / total_attr_labels
    return average_recall, average_percision

'''precision is the fraction of events where we correctly declared 𝑖 / instances where the algorithm declared 𝑖.
recall is the fraction of events where we correctly declared 𝑖 / of the cases where the true of state of the world is 𝑖.
'''
def calcuate_polarity_recall_precision(all_reviews):
    attributes_possible = {
        'positive', 'neutral', 'negative'
    }
    recalls_added_together = 0
    precisions_added_together = 0
    algorithm_amounts_of_attribute_type = {}
    expected_amounts_of_attribute_type = {}
    correct_amounts_of_attribute_type = {}
    for review in all_reviews:
        for sentence in review.sentences:
            for i in range(0, len(sentence.opinions_predicted)):
                opinion_pred = sentence.opinions_predicted[i]
                opinion_exp = sentence.opinions_expected[i]
                attr_pred = opinion_pred.polarity
                attr_exp = opinion_exp.polarity
                if attr_exp in expected_amounts_of_attribute_type:
                    expected_amounts_of_attribute_type[attr_exp] += 1
                else:
                    expected_amounts_of_attribute_type[attr_exp] = 1
                if attr_pred in algorithm_amounts_of_attribute_type:
                    algorithm_amounts_of_attribute_type[attr_pred] += 1
                else:
                    algorithm_amounts_of_attribute_type[attr_pred] = 1
                if attr_exp == attr_pred:
                    if attr_pred in correct_amounts_of_attribute_type:
                        correct_amounts_of_attribute_type[attr_pred] += 1
                    else:
                        correct_amounts_of_attribute_type[attr_pred] = 1
    total_attr_labels = len(correct_amounts_of_attribute_type)
    for attr_type in attributes_possible:
        if attr_type in correct_amounts_of_attribute_type:
            recall = correct_amounts_of_attribute_type[attr_type] / expected_amounts_of_attribute_type[attr_type]
            recalls_added_together += recall
            percision = correct_amounts_of_attribute_type[attr_type] / algorithm_amounts_of_attribute_type[attr_type]
            precisions_added_together += percision
    if total_attr_labels == 0:
        return 0.0, 0.0
    average_percision = precisions_added_together / total_attr_labels
    average_recall = recalls_added_together / total_attr_labels
    return average_recall, average_percision











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
    all_reviews = parse_xml(path_train, opinion_expected, stop_words)
    process_reviews(all_reviews, stop_words)
