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

category_dict = {'RESTAURANT#GENERAL': {'NULL': '', 'trattoria': '', 'restaurant': '', 'place': '', 'Saul': '', 'Fish': '', 'Restaurant Saul': '', 'pink pony': '', 'spot': '', 'Leon': '', 'Zucchero Pomodori': '', 'Gnocchi': '', 'Planet Thailand': '', 'Mizu': '', 'Jekyll and Hyde': '', 'Cafe Spice': '', 'Red Eye': '', 'Red Eye Grill': '', 'Jekyll and hyde Pub': '', 'bar': '', 'Jekyll and Hyde Pub': '', 'fresh restaurant': '', 'Big Wong': '', 'Patis': '', 'Pastis': '', 'Teodora': '', 'Myagi': '', 'Prune': '', 'Jeckll and Hydes': '', 'Shabu-Shabu Restaurant': '', 'Emilio': '', 'Amma': '', 'Haru on Park S': '', "Roth's": '', 'Planet Thai': '', 'Chennai Garden': '', 'PLACE': '', 'YUKA': '', 'Mermaid Inn': '', 'Cafe Noir': '', 'Casimir': '', 'mare': '', 'pizzeria': '', 'joint': '', 'pizza place': '', 'PIZZA 33': '', 'Pizza 33': '', 'Williamsburg spot': '', 'Lucky Strike': '', 'Suan': '', 'Ginger House': '', 'Chinese restaurant': '', 'Rao': '', 'Heartland Brewery': '', 'Corona': '', "Bloom's": '', 'Faan': '', "Rao's": '', 'Areo': '', 'Al Di La': '', 'Cypriot restaurant': '', 'restaraunt': '', 'The Four Seasons': '', 'Yamato': '', 'Indian Restaurant': '', "Water's Edge": '', 'Casa La Femme': '', "Patsy's Pizza": '', 'Village Underground': '', 'establishment': '', 'Bukhara Grill': '', 'Bukhara': '', 'east village pizza': '', 'modern Japanese brasserie': '', 'brasserie': '', 'Zenkichi': '', 'Pacifico': '', 'palce': '', 'Restaurant': '', 'Bark': '', 'The Four Seasons restaurant': '', 'Casa la Femme': ''}, 'RESTAURANT#PRICES': {'place': '', 'NULL': '', 'resturant': '', 'restaurant': '', 'Suan': '', 'Rice Avenue': '', "Baluchi's": '', 'The Four Seasons': '', 'location': '', 'Casa La Femme': '', 'brasserie': '', 'Bark': ''}, 'RESTAURANT#MISCELLANEOUS': {'NULL': '', 'crowd': '', 'place': '', 'signs': '', "VT's": '', '1st Ave spot': '', 'owner': '', 'staff': '', 'restaurant': '', 'spot': '', 'Thai restaurant': '', 'pizza place': '', 'sushi chef': '', 'establishment': '', 'Place': '', 'space': '', 'indian place': '', 'Pacifico': '', "Water's Edge": '', 'location': '', 'Yamato': '', 'The Four Seasons': '', 'Casa La Femme': '', 'Bark': '', 'BFC': ''}, 'FOOD#PRICES': {'food': '', 'meal': '', 'half price sushi deal': '', 'Prix Fixe menu': '', 'NULL': '', 'pizza': '', 'toppings': '', 'dessert': '', 'sushi': '', 'spicy Tuna roll': '', 'all you can eat deal': '', 'congee': '', 'noodles': '', 'rice dishes': '', 'prixe fixe tasting menu': '', 'dishes': '', 'sandwiches': '', 'Indian': '', 'seafood': '', 'lobster sandwich': '', 'Cheese plate': '', 'pizzas': '', 'menu': '', 'Taiwanese food': '', 'Food': '', 'dim sum': '', 'Spreads': '', 'eats': '', 'Chinese food': '', 'fish': '', 'all you can eat sushi': '', 'fish and chips': '', 'japanese food': '', 'dinner': '', 'guacamole': '', 'dish': '', 'dinner for two': '', 'chicken tikka masala': '', 'lunch buffet': '', 'stone bowl': '', 'BBE $29 fixe prix menu': '', 'pita bread': '', 'kimchi': '', 'salad': '', 'salmon': '', 'eggplant': ''}, 'FOOD#QUALITY': {'food': '', 'lava cake dessert': '', 'Pizza': '', 'Salads': '', 'calamari': '', 'NULL': '', 'good': '', 'Guacamole+shrimp appetizer': '', 'filet': '', 'frites': '', 'dishes': '', 'specials': '', 'regular menu-fare': '', 'parmesean porcini souffle': '', 'lamb glazed with balsamic vinegar': '', 'pad se ew chicken': '', 'pad thai': '', 'Food': '', 'Chow fun': '', 'pork shu mai': '', 'meal': '', 'pizza': '', 'foie gras terrine with figs': '', 'duck confit': '', 'oysters': '', 'duck breast special': '', 'Thai fusion stuff': '', 'Grilled Chicken special with Edamame Puree': '', 'Edamame pureed': '', 'spicy tuna roll': '', 'rock shrimp tempura': '', 'sea urchin': '', 'sushi': '', 'half price sushi deal': '', 'Prix Fixe menu': '', 'somosas': '', 'chai': '', 'chole': '', 'dhosas': '', 'dhal': '', 'French Onion soup': '', 'desserts': '', 'cheese': '', 'ingredients': '', 'crust': '', 'meals': '', 'seafood': '', 'Pastrami': '', 'fried shrimp': '', 'French bistro fare': '', 'lunch': '', 'Sauce': '', 'tuna of gari': '', 'thai food': '', 'rolls': '', 'sashimi': '', 'crunchy tuna': '', 'French food': '', 'Spicy Scallop roll': '', 'Moules': '', 'lobster ravioli': '', "chef's specials": '', 'exotic food': '', 'tuna': '', 'wasabe potatoes': '', 'fresh mozzarella': '', 'pie': '', 'salad': '', 'dining': '', 'chicken pot pie': '', 'cheeseburger': '', 'bagels': '', 'Uni Hand roll': '', 'lobster teriyaki': '', 'rose special roll': '', 'pork belly': '', 'raw vegatables in side orders': '', 'balance of herbs and tomatoes': '', 'pumkin tortelini': '', 'lobster roll': '', 'lobster': '', 'santa fe chopped salad': '', 'fish and chips': '', 'chow fun and chow see': '', 'scallion pancakes': '', 'fried dumplings': '', 'pad penang': '', 'chef': '', 'salads': '', "Pam's special fried fish": '', 'Ingredients': '', 'spicy Tuna roll': '', 'Yellowtail': '', 'Caesar Salad': '', 'arugula and goat cheese': '', 'pasta dish': '', 'tiramisu': '', 'chocolate cake': '', 'raddichio': '', 'mushroom pizza': '', 'homemade pasta': '', 'hanger steak': '', 'filet mignon dish': '', 'beef and noodle soup dishes': '', 'rosemary or orange flavoring': '', 'Fish': '', 'dessert': '', 'fish': '', 'tuna tartar appetizer': '', 'New England Chowder': '', 'Lobster Bisque': '', 'Prime Rib': '', 'chicken vindaloo': '', "Chef's tasting menu": '', 'prixe fixe tasting menu': '', 'lemon salad': '', 'grilled branzino': '', 'bagel': '', 'lox': '', 'Shabu-Shabu': '', 'Taxan': '', 'green curry with vegetables': '', 'ravioli': '', 'marinara/arrabiatta sauce': '', 'mozzarella en Carozza': '', 'pepperoni': '', 'family style salad': '', 'vegetarian dishes': '', 'non-veg selections': '', 'sea bass': '', 'Dal Bukhara': '', 'kababs': '', 'rice': '', 'all-u-can-eat sushi': '', 'soy sauce': '', 'dinner': '', 'eggs benedict': '', 'Pad Thai': '', 'Indian': '', 'lobster sandwich': '', 'spaghetti with Scallops and Shrimp': '', 'halibut special': '', 'steak': '', 'foods': '', 'jelly fish': '', 'drunken chicken': '', 'soupy dumplings': '', 'stir fry blue crab': '', 'Cheese plate': '', 'asparagus, truffle oil, parmesan bruschetta': '', 'thai cuisine': '', 'caviar': '', 'salmon dish': '', 'dim sum': '', 'cheesecake': '', 'pastries': '', 'spice': '', 'Tom Kha soup': '', 'Thai': '', 'pesto pizza': '', 'spicy Italian cheese': '', 'french fries': '', 'scallops': '', 'sauce': '', 'japanese comfort food': '', 'lamb sausages': '', 'sardines with biscuits': '', 'large whole shrimp': '', 'pistachio ice cream': '', 'ceviche mix (special)': '', 'crab dumplings': '', 'assorted sashimi': '', 'banana tempura': '', 'Thai food': '', 'Gulab Jamun (dessert)': '', 'pizzas': '', 'Sophia pizza': '', 'kitchen food': '', 'Sushi': '', 'cuisine': '', 'smoked salmon and roe appetizer': '', 'entree': '', 'menu': '', 'Taiwanese food': '', 'cold appetizer dishes': '', 'mahi mahi (on saffron risotto': '', 'chicken and mashed potatos': '', 'crab cakes': '', 'selection of thin crust pizza': '', 'Basil slice': '', 'calzones': '', 'dosas': '', 'sandwiches': '', 'Italian food': '', 'basic dishes': '', 'apppetizers': '', 'sushimi cucumber roll': '', 'spreads': '', 'cream cheeses': '', 'Bagels': '', 'turkey burgers': '', 'Japanese food': '', 'soup for the udon': '', 'Japanese cuisine': '', 'Margheritta slice': '', 'appetizer menu': '', 'brioche and lollies': '', 'salmon': '', 'crab salad': '', 'mussels in spicy tomato sauce': '', 'fries': '', 'noodles with shrimp and chicken and coconut juice': '', 'Indian food': '', 'balsamic vinegar over icecream': '', 'Go Go Hamburgers': '', 'turnip cake': '', 'roast pork buns': '', 'egg custards': '', 'braised lamb shank in red wine': '', 'Spreads': '', 'toppings': '', 'indian cuisine': '', 'shrimp appetizers': '', 'eats': '', 'indian food': '', 'baked clams octopus': '', 'lamb': '', 'Appetizers': '', 'potato stuff kanish': '', 'chicken': '', 'Dessert': '', 'anti-pasta': '', 'pasta mains': '', 'shrimp scampi': '', 'porcini mushroom pasta special': '', 'seafood tagliatelle': '', 'BBQ ribs': '', 'rice dishes': '', 'congee (rice porridge)': '', 'hot sauce': '', 'cheescake': '', 'chicken casserole': '', 'beef': '', 'lamb dishes': '', 'Reuben sandwich': '', 'sauces': '', 'Ravioli': '', 'Pakistani food': '', 'mussles': '', 'seabass': '', 'goat cheese salad': '', 'penne w/ chicken': '', 'desert': '', 'roti rolls': '', 'Unda (Egg) rolls': '', 'spices': '', 'onions': '', 'eggs': '', 'roti': '', 'drumsticks over rice': '', 'sour spicy soup': '', 'Beef noodle soup': '', 'dumplings': '', '$10 10-piece dim sum combo': '', 'crabmeat lasagna': '', 'chocolate bread pudding': '', 'egg noodles in the beef broth with shrimp dumplings and slices of BBQ roast pork': '', 'dish': '', 'congee': '', 'Ow Ley Soh': '', 'Chinese food': '', 'Japanese Tapas': '', 'Yakitori (bbq meats)': '', 'nigiri': '', 'pasta penne': '', 'La Rosa': '', 'mussels': '', 'Thin Crust Pizzas': '', 'Lasagna Menu': '', 'BBQ Salmon': '', 'Sea Bass': '', 'Crispy Duck': '', 'pastas': '', 'risottos': '', 'sepia': '', 'braised rabbit': '', 'Dog': '', 'dog': '', 'pork souvlaki': '', 'eggplant pizza': '', 'millennium roll': '', 'seafood spaghetti': '', 'indo-chinese food': '', 'chicken pasta': '', 'vitello alla marsala': '', 'veal': '', 'mushrooms': '', 'potato balls': '', 'Red Dragon Roll': '', 'Seafood Dynamite': '', 'japanese food': '', 'Dancing, White River and Millenium rolls': '', 'quesadilla': '', 'guacamole': '', 'Indian Food': '', 'baba ganoush': '', 'omlette for brunch': '', 'spinach': '', 'quacamole': '', 'wings with chimmichuri': '', 'chicken in the salads': '', 'portobello and asparagus mole': '', 'gyros': '', 'gyro meat': '', 'sausages': '', 'Greek and Cypriot dishes': '', 'gyro': '', 'stuff tilapia': '', 'bread': '', 'appetizer of olives': '', 'main course': '', 'pear torte': '', 'dogs': '', 'hot dog': '', 'mushroom sauce': '', 'triple color and norwegetan rolls': '', 'banana chocolate dessert': '', 'green tea tempura': '', 'appetizers': '', 'modern Japanese': '', 'modern Japanese food': '', 'Indo Chinese food': '', 'Chinese style Indian food': '', 'chicken lollipop': '', 'Chilli Chicken': '', 'vegetarian dish': '', 'hot dogs': '', 'indian chinese food': '', 'Indian Chinese': '', 'Vanison': '', 'Bison': '', 'dessserts': '', 'fried oysters and clams': '', 'lobster knuckles': '', 'Thai style Fried Sea Bass': '', 'grilled Mahi Mahi': '', 'lunch buffet': '', 'kimchee': '', 'Korean fair': '', 'four course prix fix menu': '', 'bibimbap': '', 'nakgi-bokum': '', 'stir-fried squid': '', 'side dishes': '', 'risotto': '', 'farro salad': '', 'mashed yukon potatoes': '', 'margherita pizza': '', 'slice of NYC pizza': '', 'sashimi amuse bouche': '', 'Grilled Black Cod': '', 'Grilled Salmon dish': '', 'frozen black sesame mousse': '', 'matcha (powdered green tea) and blueberry cheesecake': '', 'Shabu Shabu': '', 'meat': '', 'Korean food': '', 'fusion twists': '', 'pork belly tacos': '', 'pork croquette sandwich': '', 'bun': '', 'family seafood entree': '', 'main entree': '', 'appetizer': '', 'pita': '', 'hummus': '', 'grilled octopus': '', 'eggplant': '', 'Hot Dogs': '', 'Slamwich': '', 'fish dishes': '', 'Mussles': '', 'Lamb special': '', 'flank steak': '', 'fish tacos': '', 'pasta': ''}, 'FOOD#STYLE_OPTIONS': {'portions': '', 'Edamame pureed': '', 'sushi': '', 'rice to fish ration': '', 'Prix Fixe menu': '', 'food': '', 'French Onion soup': '', 'NULL': '', 'menu': '', 'portion': '', 'fried shrimp': '', 'specials menus': '', 'rolls': '', 'Steak Tartare': '', 'dessert': '', 'exotic food': '', 'appetizer selection': '', 'cheeseburger': '', 'bagel': '', 'spicy Tuna roll': '', 'all you can eat deal': '', 'dishes': '', 'Lobster Bisque': '', 'bagels': '', 'cream cheeses': '', 'fish': '', 'rice': '', 'sandwiches': '', 'portion sizes': '', 'lobster sandwich': '', 'Cheese plate': '', 'bruschettas': '', 'paninis': '', 'tramezzinis': '', 'buffet': '', 'cheeseburgers': '', 'burgers': '', 'pastrami sandwich on a roll': '', 'entree': '', 'selection of thin crust pizza': '', 'dosas': '', 'cheff': '', 'servings for main entree': '', 'veal': '', 'pasta mains': '', 'antipasti': '', 'BBQ ribs': '', 'chicken casserole': '', 'pastas': '', 'Personal pans': '', 'pizza': '', 'penne a la vodka': '', 'pasta penne': '', 'selection': '', 'half/half pizza': '', 'Chicken teriyaki': '', 'japanese food': '', 'quesadilla': '', 'toppings': '', 'trimmings': '', 'triple color and norwegetan rolls': '', 'special roll': '', 'regular roll': '', 'vegetarian dish': '', 'fried oysters and clams': '', 'lobster knuckles': '', '"salt encrusted shrimp" appetizer': '', 'grilled Mahi Mahi': '', 'Indian food': '', 'stone bowl': '', 'side dishes': '', 'BBE $29 fixe prix menu': '', 'meal': '', 'japanese tapas': '', 'chicken': '', 'hot dog': '', 'salad': '', 'pasta': ''}, 'DRINKS#PRICES': {'wine list': '', 'wine': '', 'house champagne': '', 'wine selection': '', 'wines': '', 'Drinks': '', 'Wine list': '', 'bottle minimun': '', 'martini': '', 'selecion of wines': '', 'bottles of wine': '', 'drinks': '', 'Voss bottles of water': '', 'bev': '', 'martinis': ''}, 'DRINKS#QUALITY': {'sake': '', 'wine': '', 'glass of wine': '', 'Gigondas': '', 'drinks': '', 'sangria': '', 'wines': '', 'expresso': '', 'iced tea': '', 'NULL': '', 'Change Mojito': '', 'martinis': '', 'premium sake': '', 'strawberry daiquiries': '', 'coffee': '', 'martini': '', 'Vanilla Shanty': '', 'SEASONAL beer': '', 'beer': '', 'drink': '', 'bottles of wine': '', 'sassy lassi': '', 'wine by the glass': '', 'margaritas': '', 'bar drinks': '', 'cocktail with Citrus Vodka and lemon and lime juice and mint leaves': '', 'sake’s': '', 'bottle of wine': ''}, 'DRINKS#STYLE_OPTIONS': {'Bombay beer': '', 'wine list': '', 'sake list': '', 'wine selection': '', 'bar': '', 'beers': '', 'sake menu': '', 'Wine list selection': '', 'wine-by-the-glass': '', 'wine': '', 'bottles of Korbett': '', 'selection of wines': '', 'wine choices': '', 'Wine list': '', 'beverage selections': '', 'martini': '', 'measures of liquers': '', 'beer': '', 'selecion of wines': '', 'wines by the glass': '', 'drink menu': '', 'selection of bottled beer': ''}, 'AMBIENCE#GENERAL': {'Decor': '', 'place': '', 'trattoria': '', 'candle-light': '', 'tables': '', 'interior decor': '', 'interior': '', 'space': '', 'decor': '', 'vent': '', 'Ambiance': '', 'NULL': '', 'ambience': '', 'Cosette': '', 'restaurant': '', 'Leon': '', 'atmosphere': '', 'atmoshpere': '', 'garden terrace': '', 'open kitchen': '', 'vibe': '', 'setting': '', 'Downstairs lounge': '', "Raga's": '', 'shows': '', 'actors': '', 'Traditional French decour': '', 'hall': '', 'semi-private boths': '', 'live jazz band': '', 'hidden bathrooms': '', 'Ambience': '', 'ambient': '', 'backyard dining area': '', 'dining room': '', 'ambiance': '', 'characters': '', 'resturant': '', 'feel': '', 'back room': '', 'room': '', 'seats': '', 'cigar bar': '', 'mileau': '', 'outside table': '', 'unisex bathroom': '', 'back patio': '', 'music': '', 'back garden sitting area': '', 'blond wood decor': '', 'Thalia': '', 'outdoor atmosphere': '', 'garden': '', 'atmoshere': '', 'late night atmosphere': '', 'in-house lady DJ': '', 'terrace': '', 'bar scene': '', 'bar': '', 'atomosphere': '', 'main dining room': '', 'ceiling': '', 'patio': '', 'outdoor seating': '', 'setting/atmosphere': '', 'design': '', 'jukebox': '', 'Toons': '', 'Indoor': '', 'jazz duo': '', 'Atmosphere': '', 'Rice Avenue': '', 'scene': '', 'Dining Garden': '', 'Jazz Bar': '', 'spot': '', 'back garden area': '', 'Egyptian restaurant': '', 'belly dancers': '', 'hookah': '', 'booths': '', 'rooms': '', 'bathroom': '', 'mens bathroom': '', 'seating': '', 'boths': '', 'looks': '', 'belly dancing show': '', 'scheme of mirrors': '', 'mirrors': '', 'DJ': '', 'environment': '', 'white organza tent': '', 'Restaurant': '', 'modern Japanese brasserie': '', 'unmarked wooden doors': '', 'décor': '', 'private booths': '', 'glass ceilings': '', 'Zenkichi': '', 'mirrored walls': '', 'sitting space': '', 'fit-out': '', 'furnishings': '', 'fire place': ''}, 'SERVICE#GENERAL': {'service': '', 'Service': '', 'people': '', 'cart attendant': '', 'NULL': '', 'staff': '', 'waiter': '', 'kitchen': '', 'waitstaff': '', 'hostess': '', 'wait': '', 'waitress': '', "maitre d'": '', 'server': '', 'Wait staff': '', 'Seating': '', 'waiters': '', 'takeout': '', 'Manager': '', 'wait staff': '', 'servers': '', 'Delivery': '', 'seating': '', 'waitstaffs': '', 'svc': '', 'Waitstaff': '', 'manager': '', 'customer service': '', 'gentleman': '', 'delivery': '', 'counter service': '', 'clerks': '', 'proprietor': '', 'Winnie': '', 'management': '', 'People': '', 'owner': '', 'Usha': '', 'Staff': '', 'wait-staff': '', 'delivery guys': '', 'Vittorio': '', 'crew': '', 'Delivery guy': '', 'bartender': '', 'waitresses': '', 'survice': '', 'Raymond': '', 'Paul': '', 'service button': '', 'Greg': '', 'runner': '', 'chef': '', 'SERVICE': '', 'STAFF': '', 'front of house staff': '', 'girl': '', 'Maitre-D': '', 'maitre-D': '', 'frontman': ''}, 'LOCATION#GENERAL': {'view': '', 'block': '', 'neighborhood': '', 'outdoor atmosphere': '', 'NULL': '', 'location': '', 'restaurant': '', 'place': '', 'Rice Avenue': '', 'view of the new york city skiline': '', 'views of the city': '', 'view of river and NYC': '', 'views': '', 'spot': ''}}


stanza.download('en')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
lemmatizer = WordNetLemmatizer()
nlp = stanza.Pipeline('en')





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
                    category_dict[category][target] = ''
                    opinion_expected = Opinion(sentence_id, review_id, target, target_begin_index, target_end_index, category, polarity)
                    opinions_expected.append(opinion_expected)
                    #logic to parse our own opinion result
                '''
                total_exp_opinions = len(opinions_expected)
                opinions_predicted_total = predict_opinion(sentence_id, review_id, text, total_exp_opinions)
                for opinion_pred_said in opinions_predicted_total:
                    opinions_predicted.append(opinion_pred_said)
                '''
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
            sentence_gathered = Sentence(sentence_id, review_id, text, opinions_predicted, opinions_expected)
            sentences_for_review_object.append(sentence_gathered)
        review_collected = Review(review_id, sentences_for_review_object)
        all_reviews.append(review_collected)
    for review in all_reviews:
        for sentence in review.sentences:
            total_exp_opinions = len(sentence.opinions_expected)
            opinions_predicted_total = predict_opinion(sentence.sentence_id, sentence.review_id, sentence.text, total_exp_opinions)
            for opinion_pred_said in opinions_predicted_total:
                sentence.opinions_predicted.append(opinion_pred_said)
    print(category_dict)
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

def predict_opinion(sentence_id, review_id, text, total_exp_opinions):
    target_pred, ote_with_emotion_adj = extract_opinion_target(text, total_exp_opinions)
    category_pred = label_opinion_category(text, target_pred, total_exp_opinions)
    polarity_pred = label_opinion_polarity(text, target_pred, category_pred, ote_with_emotion_adj, total_exp_opinions)
    opinions = []
    for i in range(total_exp_opinions):
        opinion_pred = Opinion(sentence_id, review_id, target_pred[i][0], target_pred[i][1], target_pred[i][2], category_pred[i], polarity_pred[i])
        opinions.append(opinion_pred)
    return opinions

def extract_opinion_target(text, total_exp_opinions):
    stop_words = set(stopwords.words('english'))
    sentence_given = text.lower()
    sentence_given_words = text.split()
    sentence_tokenized = nltk.sent_tokenize(sentence_given)
    for sentence in sentence_tokenized:
        tokenized_words_in_sentence = nltk.word_tokenize(sentence)
        first_pos_tags_on_words_in_sentence = nltk.pos_tag(tokenized_words_in_sentence)
    part_two_of_target = False
    part_three_of_target = 0
    new_word_list = []
    pos_tags_on_words_in_sentence = []
    for i in range(0, len(first_pos_tags_on_words_in_sentence)-1):
        if first_pos_tags_on_words_in_sentence[i][1] == 'NN' and first_pos_tags_on_words_in_sentence[i+1][1] == 'NN':
            if i < len(first_pos_tags_on_words_in_sentence)-2 and first_pos_tags_on_words_in_sentence[i+1][1] == 'NN' and first_pos_tags_on_words_in_sentence[i+2][1] == 'NN' and part_three_of_target == 0 and part_two_of_target == False:
                pos_tags_on_words_in_sentence.append((first_pos_tags_on_words_in_sentence[i][0] + ' ' + first_pos_tags_on_words_in_sentence[i+1][0] + ' ' + first_pos_tags_on_words_in_sentence[i+2][0], first_pos_tags_on_words_in_sentence[i+1][1]))
                part_three_of_target += 1
                new_word_list.append(first_pos_tags_on_words_in_sentence[i][0] + first_pos_tags_on_words_in_sentence[i+1][0] + first_pos_tags_on_words_in_sentence[i+2][0])
            else: 
                pos_tags_on_words_in_sentence.append((first_pos_tags_on_words_in_sentence[i][0] + ' ' + first_pos_tags_on_words_in_sentence[i+1][0], first_pos_tags_on_words_in_sentence[i+1][1]))
                part_two_of_target = True
                new_word_list.append(first_pos_tags_on_words_in_sentence[i][0] + first_pos_tags_on_words_in_sentence[i+1][0])
        else:
            if part_two_of_target:
                part_two_of_target = False
                continue
            if part_three_of_target > 0 and part_three_of_target < 3:
                part_three_of_target += 1
                if part_three_of_target == 3:
                    part_three_of_target = 0
                continue
            if first_pos_tags_on_words_in_sentence[i][0] in stop_words:
                new_word_list.append(first_pos_tags_on_words_in_sentence[i][0])
                continue
            new_word_list.append(first_pos_tags_on_words_in_sentence[i][0])
            pos_tags_on_words_in_sentence.append(first_pos_tags_on_words_in_sentence[i])
    #UNCOMMENT OUT to see the pos tags on the word in the sentence that are not stop words
    '''
    print(pos_tags_on_words_in_sentence)
    '''
    updated_sentence = ' '.join(new_word_list)
    updated_sentence_words_tokenized = nltk.word_tokenize(updated_sentence)
    updated_sentence_words_list = [word for word in updated_sentence_words_tokenized if not word in stop_words]
    updated_pos_tags_on_words_in_sentence = nltk.pos_tag(updated_sentence_words_list)
    doc = nlp(updated_sentence)
    dependency_node = []
    #print(text)
    for dependency_edge in doc.sentences[0].dependencies:
        dependency_node.append([dependency_edge[2].text, dependency_edge[0].id, dependency_edge[1]])
    #print(dependency_node)
    for i in range(0, len(dependency_node)):
        if (int(dependency_node[i][1]) != 0):
            #print(new_word_list)
            #print(int(dependency_node[i][1])-1)
            dependency_node[i][1] = dependency_node[(int(dependency_node[i][1])-1)][0]
    target_list = []
    for part in updated_pos_tags_on_words_in_sentence:
        pos = part[1]
        word = part[0]
        if (pos == 'NN' or pos == 'JJ' or pos == 'JJR' or pos == 'NNS' or pos == 'RB'):
            target_list.append(list(part))
    feature_cluster = []
    for target in target_list:
        f_target_list = []
        for j in dependency_node:
            '''
            print(f'J[0]: {j[0]}')
            print(f'TARGET[0]: {target[0]}')
            print(f'J[1]: {j[1]}')
            '''
            if ((target[0].find(str(j[0])) != -1 or target[0].find(str(j[1])) != -1) and (j[2] in ["nsubj", "acl:relcl", "obj", "dobj", "agent", "advmod", "amod", "neg", "prep_of", "acomp", "xcomp", "compound"])):
                if j[0] == target[0]:
                    f_target_list.append(j[1])
                else:
                    f_target_list.append(j[0])
        feature_cluster.append([target[0], f_target_list])
    #print(feature_cluster)
    ote_with_emotion_adj = []
    vocab = {}
    for target in target_list:
        #print(target)
        vocab[target[0]] = target[1]
    for target in feature_cluster:
        if vocab[target[0]] == 'NN' or vocab[target[0]] == 'NNS' or vocab[target[0]] == 'NNP':
            ote_with_emotion_adj.append(target)
    for cluster in ote_with_emotion_adj:
        for pos_tag_word in pos_tags_on_words_in_sentence:
            pos_tag_word_list = pos_tag_word[0].split()
            one_word_combined = ''.join(pos_tag_word_list)
            if cluster[0] == one_word_combined and len(pos_tag_word_list) > 1:
                cluster[0] = pos_tag_word[0]
    #print(ote_with_emotion_adj)
    #TODO what about Nulls?
    targets_to_return = []
    for i in range(0, total_exp_opinions):
        if len(ote_with_emotion_adj) == 0:
            targets_to_return.append(['NULL', "0", "0"])
        elif i < len(ote_with_emotion_adj)-1:
            #TODO just 0s?
            targets_to_return.append([ote_with_emotion_adj[i][0], "0", "0"])
        elif i >= len(ote_with_emotion_adj)-1:
            if i == total_exp_opinions - 1 and i == len(ote_with_emotion_adj)-1:
                targets_to_return.append([ote_with_emotion_adj[i][0], "0", "0"])
            elif i <= total_exp_opinions - 1:
                target_compiled = ''
                for c in ote_with_emotion_adj:
                    target_compiled = c[0] + ' '
                targets_to_return.append([target_compiled, "0", "0"])
    return targets_to_return, ote_with_emotion_adj
    #return "BASIC", "0", "0"

def label_opinion_category(text, target_pred, total_exp_opinions):
    entities = label_opinion_entity(text, target_pred, total_exp_opinions)
    attributes = []
    categories = []
    for i in range(0, total_exp_opinions):
        for key in category_dict:
            if target_pred[i][0] in category_dict[key]:
                categories.append(key)
            else:
                categories.append('NULL#NULL')
    return categories
    #return "BASIC#BASIC"

def label_opinion_polarity(text, target_pred, category_pred, ote_with_emotion_adj, total_exp_opinions):
    polarities = []
    for i in range(0, total_exp_opinions):
        polarity_scores = []
        if len(ote_with_emotion_adj) == 0:
            polarities.append('neutral')
        elif i < len(ote_with_emotion_adj) - 1:
            feeling_words = ote_with_emotion_adj[i][1]
            feeling_adj = ' '.join(feeling_words)
            feeling_tokenized = nltk.sent_tokenize(feeling_adj)
            if len(feeling_tokenized) > 0:
                for sentence in feeling_tokenized:
                    tokenized_words_in_sentence = nltk.word_tokenize(sentence)
                    pos_tags_on_feeling_words = nltk.pos_tag(tokenized_words_in_sentence)
                for tag in pos_tags_on_feeling_words:
                    if tag[1].startswith('N') or tag[1].startswith('J') or tag[1].startswith('R') or tag[1].startswith('V'):
                        pos_word = determin_pos(tag[1])
                        lemma = lemmatizer.lemmatize(tag[0], pos=pos_word)
                        if not lemma:
                            polarity_scores.append([])
                        else:
                            synsets = wordnet.synsets(tag[0], pos=pos_word)
                            if not synsets:
                                polarity_scores.append([])
                            if len(synsets) == 0:
                                polarity_scores.append([])
                            else:
                                synset = synsets[0]
                                swn_synset = swn.senti_synset(synset.name())
                                polarity_scores.append([swn_synset.pos_score(), swn_synset.neg_score(), swn_synset.obj_score()])
                    else:
                        polarity_scores.append([])
            else: 
                polarity_scores.append([])
            polarities.append(determine_polarity(polarity_scores))
        elif i >= len(ote_with_emotion_adj)-1:
            if i == total_exp_opinions - 1 and i == len(ote_with_emotion_adj)-1:
                feeling_words = ote_with_emotion_adj[i][1]
                feeling_adj = ' '.join(feeling_words)
                feeling_tokenized = nltk.sent_tokenize(feeling_adj)
                if len(feeling_tokenized) > 0:
                    for sentence in feeling_tokenized: 
                        tokenized_words_in_sentence = nltk.word_tokenize(sentence)
                        pos_tags_on_feeling_words = nltk.pos_tag(tokenized_words_in_sentence)
                    for tag in pos_tags_on_feeling_words:
                        if tag[1].startswith('N') or tag[1].startswith('J') or tag[1].startswith('R') or tag[1].startswith('V'):
                            pos_word = determin_pos(tag[1])
                            lemma = lemmatizer.lemmatize(tag[0], pos=pos_word)
                            if not lemma:
                                polarity_scores.append([])
                            else:
                                synsets = wordnet.synsets(tag[0], pos=pos_word)
                                if not synsets:
                                    polarity_scores.append([])
                                if len(synsets) == 0:
                                    polarity_scores.append([])
                                else:
                                    synset = synsets[0]
                                    swn_synset = swn.senti_synset(synset.name())
                                    polarity_scores.append([swn_synset.pos_score(), swn_synset.neg_score(), swn_synset.obj_score()])
                        else:
                            polarity_scores.append([])
                else:
                    polarity_scores.append([])
                polarities.append(determine_polarity(polarity_scores))
            else:
                feeling_adj = ''
                for f_c in ote_with_emotion_adj:
                    feeling_adj += ' '.join(f_c[1])
                feeling_tokenized = nltk.sent_tokenize(feeling_adj)
                if len(feeling_tokenized) > 0:
                    for sentence in feeling_tokenized: 
                        tokenized_words_in_sentence = nltk.word_tokenize(sentence)
                        pos_tags_on_feeling_words = nltk.pos_tag(tokenized_words_in_sentence)
                    for tag in pos_tags_on_feeling_words:
                        if tag[1].startswith('N') or tag[1].startswith('J') or tag[1].startswith('R') or tag[1].startswith('V'):
                            pos_word = determin_pos(tag[1])
                            lemma = lemmatizer.lemmatize(tag[0], pos=pos_word)
                            if not lemma:
                                polarity_scores.append([])
                            else:
                                synsets = wordnet.synsets(tag[0], pos=pos_word)
                                if not synsets:
                                    polarity_scores.append([])
                                if len(synsets) == 0:
                                    polarity_scores.append([])
                                else:
                                    synset = synsets[0]
                                    swn_synset = swn.senti_synset(synset.name())
                                    polarity_scores.append([swn_synset.pos_score(), swn_synset.neg_score(), swn_synset.obj_score()])
                        else:
                            polarity_scores.append([])
                else:
                    polarity_scores.append([])
                polarities.append(determine_polarity(polarity_scores))
    return polarities
    #return "BASIC"

def label_opinion_entity(text, target_pred, total_exp_opinions):
    return []

def determine_polarity(polarity_scores):
    polarity = 0
    for polarity_score in polarity_scores:
        if polarity_score == []:
            continue
        polarity += polarity_score[0]
        polarity -= polarity_score[1]
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'

def determin_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    elif tag.startswith('V'):
        return wordnet.VERB
    return None











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
    path_trial = 'trial_data/restaurants_trial_english_sl.xml'
    path_train = 'train_data/ABSA16_Restaurants_Train_SB1_v2.xml'
    path_test = 'test_gold_data/EN_REST_SB1_TEST.xml.gold'
    opinion_expected = True
    parse_xml(path_trial, opinion_expected)
