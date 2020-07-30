import numpy as np
import matplotlib.pyplot as plt
import json
from skimage import io
from pathlib import Path
from skimage import util

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


def predict_and_save(all_indexes_list,file_name='99',out_path_indexes='data/num_generation/'):

    out_path_indexes = Path(out_path_indexes)
    out_path_indexes.mkdir(parents=True, exist_ok=True)

    cnn_agent = CNN_model()
    X_rows = []
    real_img_files = []
    for row in all_indexes_list:
        if len(row) < 1:
            continue
        temp_row_digits = []
        temp_file = []
        for digit in row:
            # print(digit.shape)
            # print(np.expand_dims(digit, axis=2).shape)
            
            try:
                temp_row_digits.append(np.expand_dims(digit, axis=2))
                image = util.img_as_ubyte(digit)
                temp_file.append(image)
                # X_rows.append(np.expand_dims(digit, axis=2))
            except:
                pass
        X_rows.append(np.array(temp_row_digits))
        real_img_files.append(np.array(temp_file))

    row_counter = 0

    for i in range(len(X_rows)): #X_row in X_rows:
        pred = cnn_agent.predict_only_X(X_rows[i])
        # print(real_img_files[i].shape)
        number = ''
        for l in range(len(pred)):
            folder_name = str(pred[l])
            finall_file_name = str(file_name)+"_"+str(row_counter)+"_"+str(l)+".png"
            output_path = Path(out_path_indexes/'{}'.format(pred[l]))
            output_path.mkdir(parents=True, exist_ok=True)
            finall_path = output_path/finall_file_name
            # print(finall_path)
            # print(real_img_files[i][l])

            io.imsave(arr=real_img_files[i][l], fname=finall_path)
        row_counter +=1

            
    # print(len(X_rows))
    # X_rows = np.array(X_rows)
    # print(X_rows.shape)
    # print(cnn_agent.predict_only_X(X_rows))ubyte