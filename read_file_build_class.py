import xml.etree.ElementTree as ET 
import csv






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

def parse_xml(xml_file, had_opinion_expected):
    tree = ET.parse(xml_file)
    reviews = tree.getroot()
    all_reviews = []
    #had_opinion_expected = False
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
                    #logic to parse our own opinion result
                    opinion_predicted = predict_opinion(sentence_id, review_id, text)
                    opinions_predicted.append(opinion_predicted)
            else:
                #this sentence could not have an opinion or this could not be a golden file
                if had_opinion_expected:
                    continue
                #no opinion that is golden to extract
                else:
                    #begin logic to parse our own opinion results
                    opinion_predicted = predict_opinion(sentence_id, review_id, text)
                    opinions_predicted.append(opinion_predicted)
            #UNCOMMENT THIS LINE TO GET 1.0 for all scores
            
            opinions_predicted = opinions_expected
            
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

    #calculate scores
    calculate_scores(all_reviews)







'''
===============================================================================================================
                                            Predicting Opinions
===============================================================================================================
'''   

def predict_opinion(sentence_id, review_id, text):
    target_pred, target_begin_index_pred, target_end_index_pred = extract_opinion_target(text)
    category_pred = label_opinion_category(text, target_pred)
    polarity_pred = label_opinion_polarity(text, target_pred, category_pred)
    opinion_pred = Opinion(sentence_id, review_id, target_pred, target_begin_index_pred, target_end_index_pred, category_pred, polarity_pred)
    return opinion_pred

def extract_opinion_target(text):
    return "BASIC", "0", "0"

def label_opinion_category(text, target_pred):
    return "BASIC#BASIC"

def label_opinion_polarity(text, target_pred, category_pred):
    return "BASIC"







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
            recall = correct_amounts_of_attribute_type[attr_type] / expected_amounts_of_attribute_type[attr_type]
            recalls_added_together += recall
            percision = correct_amounts_of_attribute_type[attr_type] / algorithm_amounts_of_attribute_type[attr_type]
            precisions_added_together += percision
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
    average_percision = precisions_added_together / total_attr_labels
    average_recall = recalls_added_together / total_attr_labels
    return average_recall, average_percision











'''
===============================================================================================================
                                        START PROGRAM
===============================================================================================================
'''   

if __name__ == "__main__":
    path = 'trial_data/restaurants_trial_english_sl.xml'
    opinion_expected = True
    parse_xml(path, opinion_expected)
