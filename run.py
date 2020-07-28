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



def main(input_dir,number_of_img,output_dir):
    


if __name__ == "__main__":

    input_dir = ''
    number_of_img = 0
    output_dir = ''


    if len(sys.argv) < 4:
        print("BRAKUJE PARAMETRU WEJŚCIOWEGO")

    print( 'Number of arguments:', len(sys.argv), 'arguments.')

    if os.path.exists(str(sys.argv[1])):
        print('Katalog z danymi', str(sys.argv[1]))
        input_dir =  str(sys.argv[1])
    else:
        print('Nie ma katalogu o podanej ścieżce: ', str(sys.argv[1]))
        exit()

    print( 'Ilośc obrazkóœ', str(sys.argv[2]))
    number_of_img = int(sys.argv[2])

    if os.path.exists(str(sys.argv[3])):
        print('Katalog z wynikami', str(sys.argv[3]))
        output_dir = str(sys.argv[3])

    else:
        print('Nie ma katalogu o podanej ścieżce: ', str(sys.argv[3]))
        exit()

    #TODO
    


    main(input_dir,number_of_img,output_dir)