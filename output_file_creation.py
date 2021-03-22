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