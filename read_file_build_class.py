import xml.etree.ElementTree as ET 
import csv

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
    def __init__(self, sentence_id, review_id, text, opinions_predicted, opinions_expected):
        self.sentence_id = sentence_id
        self.review_id = review_id
        self.text = text
        self.opinions_predicted = opinions_predicted
        self.opinions_expected = opinions_expected
    
    def print_attr(self):
        print(f'REVIEW_ID: {self.review_id}')
        print(f'SENTENCE_ID: {self.sentence_id}')
        print(f'TEXT: {self.text}')
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

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    reviews = tree.getroot()
    all_reviews = []
    had_opinion_expected = False
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
                    target = stated_opinion.attrib['target']
                    target_begin_index = stated_opinion.attrib['from']
                    target_end_index = stated_opinion.attrib['to']
                    category = stated_opinion.attrib['category']
                    polarity = stated_opinion.attrib['polarity']
                    opinion_expected = Opinion(sentence_id, review_id, target, target_begin_index, target_end_index, category, polarity)
                    opinions_expected.append(opinion_expected)
            else:
                #this sentence could not have an opinion or this could not be a golden file
                if had_opinion_expected:
                    target = 'NULL'
                    target_begin_index = '0'
                    target_end_index = '0'
                    category = 'NULL'
                    polarity = 'NULL'
                    opinion_expected = Opinion(sentence_id, review_id, target, target_begin_index, target_end_index, category, polarity)
                    opinions_expected.append(opinion_expected)
                #no opinion that is golden to extract
                else:
                    #begin logic to parse our own opinion results
                    #TODO
                    print(f'SENTENCE_ID: {sentence_id}')
            #TODO in the future remove the following line
            opinions_predicted = opinions_expected #remove when else above is implemented
            sentence_gathered = Sentence(sentence_id, review_id, text, opinions_predicted, opinions_expected)
            sentences_for_review_object.append(sentence_gathered)
        review_collected = Review(review_id, sentences_for_review_object)
        all_reviews.append(review_collected)
    #UNCOMMENT TO SEE REVIEWS
    '''
    for review in all_reviews:
        review.print_attr()
    '''
    #create .predicted and .expected files
    #TODO way to not manually change file
    target_predicted_file_name = 'output_target_data/trial.target.predicted'
    target_expected_file_name = 'output_target_data/trial.target.expected'
    create_files(target_predicted_file_name, target_expected_file_name, all_reviews, 'TARGET')

    polarity_predicted_file_name = 'output_polarity_data/trial.target.predicted'
    polarity_expected_file_name = 'output_polarity_data/trial.target.expected'
    create_files(polarity_predicted_file_name, polarity_expected_file_name, all_reviews, 'POLARITY')

    category_predicted_file_name = 'output_category_data/trial.target.predicted'
    category_expected_file_name = 'output_category_data/trial.target.expected'
    create_files(category_predicted_file_name, category_expected_file_name, all_reviews, 'CATEGORY')

def create_files(predicted_file_name, expected_file_name, all_reviews, type_file):
    predicted_file = open(predicted_file_name, "w")
    expected_file = open(expected_file_name, "w")
    for review in all_reviews:
        if type_file == 'TARGET':
            for sentence in review.sentences:
                for opinion_pred in sentence.opinions_predicted:
                    predicted_file.write(opinion_pred.string_target_attr())
                for opinion_exp in sentence.opinions_expected:
                    expected_file.write(opinion_exp.string_target_attr())
        elif type_file == 'POLARITY':
            for sentence in review.sentences:
                for opinion_pred in sentence.opinions_predicted:
                    predicted_file.write(opinion_pred.string_polarity_attr())
                for opinion_exp in sentence.opinions_expected:
                    expected_file.write(opinion_exp.string_polarity_attr())
        elif type_file == 'CATEGORY':
            for sentence in review.sentences:
                for opinion_pred in sentence.opinions_predicted:
                    predicted_file.write(opinion_pred.string_category_attr())
                for opinion_exp in sentence.opinions_expected:
                    expected_file.write(opinion_exp.string_category_attr())
    predicted_file.close()
    expected_file.close()

if __name__ == "__main__":
    path = 'trial_data/restaurants_trial_english_sl.xml'
    parse_xml(path)
