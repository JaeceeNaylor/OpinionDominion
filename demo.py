import numpy as np
import tensorflow as tf
from tensorflow import keras
import output_file_creation
import parsing
import review_opinion_sent
from nltk.corpus import stopwords
import read_file_build_class_nn

stop_words = set(stopwords.words('english'))
path_trial = 'trial_data/restaurants_trial_english_sl.xml'
path_train = 'train_data/ABSA16_Restaurants_Train_SB1_v2.xml'
path_test = 'test_gold_data/EN_REST_SB1_TEST.xml.gold'
model = keras.models.load_model('myModel')
opinion_expected = True
all_reviews = parsing.parse_xml(path_trial, opinion_expected, stop_words)

read_file_build_class_nn.process_reviews_on_predetermined_model(all_reviews, stop_words, model)

