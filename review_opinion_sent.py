import xml.etree.ElementTree as ET 
import sys
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import stanza
import re

nltk.download('stopwords')
stanza.download('en')
nlp = stanza.Pipeline('en')
food = wordnet.synset('food.n.02')
food = list(set([w for s in food.closure(lambda s:s.hyponyms()) for k in s.lemma_names() for w in k.split('_')]))
#print(food)
service = wordnet.synset('service.n.02')
service = list(set([w for s in service.closure(lambda s:s.hyponyms()) for w in s.lemma_names()]))
#print(service)
polarities = {2: 'positive', 1: 'neutral', 0: 'negative'}

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
        self.wordSemantics = []
        self.syntacticConstruction = []
        self.lexicoSemanticConstruction = []
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
            self.wordSemantics.append(0)
            self.syntacticConstruction.append(0)
            self.lexicoSemanticConstruction.append(0)
        self.headWord = ""
        self.sentPolarity = -1
        self.extract_features(text)

    def extract_features(self, text):
        doc = nlp(text)
        self.sentPolarity = int(doc.sentences[0].sentiment)
        #print(doc)
        try:
            
            for dependency_edge in doc.sentences[0].dependencies:
                #print(dependency_edge)
                if (int(dependency_edge[0].id) == 0):
                    self.headWord = dependency_edge[2].text
                else:
                    if dependency_edge[0].xpos in {'NN', 'JJ', 'JJR', 'NNS', 'RB'}:
                        if dependency_edge[1] in {"nsubj", "acl:relcl", "obj", "dobj", "agent", "advmod", "amod", "neg", "prep_of", "acomp", "xcomp", "compound"}:
                            if (int(dependency_edge[0].id) - 1 != -1):
                                if int(dependency_edge[0].id) - 1 >= len(self.syntacticConstruction):
                                    self.syntacticConstruction[len(self.syntacticConstruction) -1] = 1
                                else:
                                    self.syntacticConstruction[int(dependency_edge[0].id) - 1] = 1
                        if dependency_edge[2].xpos in {'NN', 'JJ', 'JJR', 'NNS', 'RB'}:
                            if dependency_edge[1] in {"nsubj", "acl:relcl", "obj", "dobj", "agent", "advmod", "amod", "neg", "prep_of", "acomp", "xcomp", "compound"}:
                                if (int(dependency_edge[2].id) - 1 != -1):
                                    if int(dependency_edge[2].id) - 1 >= len(self.wordSemantics):
                                        self.wordSemantics[len(self.wordSemantics) -1] = 1
                                    else:
                                        self.wordSemantics[int(dependency_edge[2].id) - 1] = 1
                    if dependency_edge[2].xpos in {'NN', 'JJ', 'JJR', 'NNS', 'RB'}:
                        if dependency_edge[1] in {"nsubj", "acl:relcl", "obj", "dobj", "agent", "advmod", "amod", "neg", "prep_of", "acomp", "xcomp", "compound"}:
                            if (int(dependency_edge[2].id) - 1 != -1):
                                if int(dependency_edge[2].id) - 1 >= len(self.syntacticConstruction):
                                    self.syntacticConstruction[len(self.syntacticConstruction) -1] = 1
                                else:
                                    self.syntacticConstruction[int(dependency_edge[2].id) - 1] = 1
        except IndexError as err:
            print("Unexpected error: {0}".format(err))
            print(text)
        #return 'Problem', 'Here'
            
        
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
    
    def setPolarity(self, sentence):
        self.polarity = polarities[sentence.sentPolarity]

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
