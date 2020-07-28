from pathlib import Path
from matplotlib import pyplot as plt
from collections import Counter
import numpy as np
import math
import sys
import os 
from os import walk,listdir
from os.path import isfile, join
import cv2
import re


from s1_finding_words_areas import get_words_from_base_img, sobel_get_img_from_background
from s2_finding_fragment_with_text import detect_fragment_with_text
from s3_finding_word import detect_fragments_with_words
from s4_snipping_digits import cut_digits_from_index_image
from s5_digit_recognition import analyze_and_predict

# TESTOWE
from skimage import io

def get_all_files_from_catalog(input_dir):
    #TODO dodaj png
    all_images = sorted(input_dir.glob("*.jpg"))
    return all_images
    

def main(input_dir,number_of_img,output_dir):
    all_images = get_all_files_from_catalog(input_dir)
    
    for image_path in all_images:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        k = re.search('[0-9]+', image_path.stem)[0]
        img_out_path_words = output_dir/"{0}-wyrazy.png".format(k)
        out_path_indexes = output_dir/"{0}-indeksy.txt".format(k)

        # print(image_path)
        # Zaciągnięcie  obrazu z pliku 
        raw_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        # Wykrycie 
        words_areas = get_words_from_base_img(raw_img.copy())
        # plt.gcf().set_size_inches(30, 20)
        # plt.imshow(words_areas,cmap = 'gray'),plt.title('??')
        # plt.show() 

        img_removed_background, reference_point_to_img_org = detect_fragment_with_text(img=words_areas, img_raw=raw_img.copy(), img_name=k)    
        
        # plt.gcf().set_size_inches(30, 20)
        # plt.imshow(img_removed_background,cmap = 'gray'),plt.title('??')
        # plt.show() 

        word_areas_from_background = sobel_get_img_from_background(img_removed_background, img_name=k)
        # plt.gcf().set_size_inches(30, 20)
        # plt.imshow(word_areas_from_background,cmap = 'gray'),plt.title('??')
        # plt.show() 

        last_word_images = detect_fragments_with_words(word_areas_from_background, raw_img.copy(), reference_point_to_img_org, img_out_path_words, img_name=k)
        
        print(len(last_word_images))
        

        # print(last_word_images[0])


        all_indexes_list = cut_digits_from_index_image(last_word_images, img_name=k)
        print("all_indexes_list: ", len(all_indexes_list))
        # plt.gcf().set_size_inches(10, 5)
        # plt.imshow(all_indexes_list[1][0],cmap = 'gray'),plt.title('??')
        # plt.show() 
        # print(len(all_indexes_list[1]))

        analyze_and_predict(all_indexes_list,out_path_indexes)

        break






if __name__ == "__main__":

    input_dir = ''
    number_of_img = 0
    output_dir = ''


    if len(sys.argv) < 4:
        print("BRAKUJE PARAMETRU WEJŚCIOWEGO")

    print( 'Number of arguments:', len(sys.argv), 'arguments.')

    if os.path.exists(str(sys.argv[1])):
        print('Katalog z danymi', str(sys.argv[1]))
        input_dir =  Path(str(sys.argv[1]))
    else:
        print('Nie ma katalogu o podanej ścieżce: ', str(sys.argv[1]))
        exit()

    print( 'Ilośc obrazkóœ', str(sys.argv[2]))
    number_of_img = int(sys.argv[2])

    if os.path.exists(str(sys.argv[3])):
        print('Katalog z wynikami', str(sys.argv[3]))
        output_dir = Path(str(sys.argv[3]))
    else:
        print('Nie ma katalogu o podanej ścieżce: ', str(sys.argv[3]))
        output_dir = Path(str(sys.argv[3]))
        output_dir.mkdir(parents=True, exist_ok=True)
        print("Utworzono katalog wyjściowy: {}".format((str(sys.argv[3]))))

    #TODO zmień parametry wejściowe
    


    main(input_dir,number_of_img,output_dir)