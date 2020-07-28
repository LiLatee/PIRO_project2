import numpy as np
import matplotlib.pyplot as plt
import json

from NN_digit_recognition import CNN_model

def analyze_and_predict(all_indexes_list,out_path_indexes):

    cnn_agent = CNN_model()
    
    words_true = []
    words_X = []
    words_y = []
    
    X_rows = []
    y_images = []


    for row in all_indexes_list:
        if len(row) < 1:
            continue
        temp_row_digits = []
        for digit in row:
            # print(digit.shape)
            # print(np.expand_dims(digit, axis=2).shape)
            
            try:
                temp_row_digits.append(np.expand_dims(digit, axis=2))
                # X_rows.append(np.expand_dims(digit, axis=2))
            except:
                pass
        X_rows.append(np.array(temp_row_digits))
    numbers_on_page = []
    for X_row in X_rows:
        pred = cnn_agent.predict_only_X(X_row)
        number = ''
        with open(out_path_indexes, 'a') as f:            
            for digit in pred:
                number += str(digit)
            f.write("%s\n" % number)
            numbers_on_page.append(number)

    print(numbers_on_page)
    # print(len(X_rows))
    # X_rows = np.array(X_rows)
    # print(X_rows.shape)
    # print(cnn_agent.predict_only_X(X_rows))
