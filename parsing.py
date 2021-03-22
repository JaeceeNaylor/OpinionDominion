import xml.etree.ElementTree as ET 
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
import review_opinion_sent

nltk.download('stopwords')



'''
===============================================================================================================
                                    PARSE XML and CREATE CLASS INSTANCES
===============================================================================================================
'''   

def parse_xml(xml_file, had_opinion_expected, stop_words):
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
                    '''used for phase 1 ro create category dict, but I understand task more now and this is not useful for other phases'''
                    #category_dict[category][target] = ''
                    opinion_expected = review_opinion_sent.Opinion(sentence_id, review_id, target, target_begin_index, target_end_index, category, polarity)
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
            #UNCOMMENT THIS LINE TO GET 1.0 for all scores
            '''
            opinions_predicted = opinions_expected
            '''
            sentence_gathered = review_opinion_sent.Sentence(sentence_id, review_id, text, opinions_predicted, opinions_expected, stop_words)
            sentences_for_review_object.append(sentence_gathered)
        review_collected = review_opinion_sent.Review(review_id, sentences_for_review_object)
        all_reviews.append(review_collected)
    return all_reviews