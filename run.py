from pathlib import Path
from skimage import io
from skimage.io import find_available_plugins
from matplotlib import pyplot as plt
import PyQt5
from skimage.filters import threshold_otsu, threshold_local, median
from skimage.color import rgb2gray, gray2rgb
from skimage.util import img_as_ubyte, img_as_bool
from skimage.filters import gaussian, laplace
from skimage.morphology import erosion, dilation, opening, closing
from skimage.exposure import rescale_intensity, match_histograms
from skimage import feature
from collections import Counter
import numpy as np
from skimage.morphology import square
from skimage.measure import approximate_polygon, subdivide_polygon, find_contours
from skimage import draw
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line
from skimage import filters
from skimage import measure
from skimage import transform
from skimage import exposure
import math
from skimage import util 
from skimage import morphology
from skimage.measure import label, regionprops
import sys
import os 
from os import walk,listdir
from os.path import isfile, join
import cv2


from s1_finding_words_areas import get_words_from_base_img, sobel_get_img_from_background
from s2_finding_fragment_with_text import detect_fragment_with_text
from s3_finding_word import detect_fragments_with_words
from s4_snipping_digits import cut_digits_from_index_image


def get_all_files_from_catalog(input_dir):
    #TODO dodaj png
    all_images = sorted(input_dir.glob("*.jpg"))
    return all_images
    

def main(input_dir,number_of_img,output_dir):
    all_images = get_all_files_from_catalog(input_dir)
    
    for image_path in all_images:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        img_out_path_words = output_dir/"{0}-wyrazy.png".format(image_name)
        out_path_indexes = output_dir/"{0}-indeksy.txt".format(image_name)
        # Zaciągnięcie  obrazu z pliku 
        raw_img = cv2.imread(str(image_path),0)

        # Wykrycie 
        words_areas = get_words_from_base_img(raw_img.copy())
        # plt.gcf().set_size_inches(30, 20)
        # plt.imshow(words_areas,cmap = 'gray'),plt.title('??')
        # plt.show() 

        img_removed_background, reference_point_to_img_org = detect_fragment_with_text(words_areas, raw_img.copy())    
        

        word_areas_from_background = sobel_get_img_from_background(img_removed_background)

        last_word_images = detect_fragments_with_words(word_areas_from_background,raw_img.copy(),reference_point_to_img_org,img_out_path_words)
        
        print(len(last_word_images[0]))

        print(last_word_images[0])


        all_indexes_list = cut_digits_from_index_image(last_word_images)

        len(all_indexes_list)
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