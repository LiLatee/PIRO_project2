import numpy as np
from pathlib import Path
from skimage import io
from skimage import measure
from skimage import morphology
from matplotlib import pyplot as plt
from skimage import util
from skimage import transform
from skimage import filters

# TESTOWE
from skimage import io
from collections import Counter

def resize_after_rotate(img_rotated, destination_shape):
    """
        Stara się wrócić do poprzedniego rozmiaru obrazu, który został przekształcony przez rotacje.

        Parameters:
        img_rotated: Przekształcony obraz, na którym dwukrotnie wykonano operacje rotacji o kąt X oraz -X.
        destination_shape: Wymiary obrazu przed obydwoma rotacjami.
    """

    diff = (np.array(img_rotated.shape) - np.array(destination_shape))/2
    if diff[0] == 0.0:
        new = img_rotated[int(diff[0]):, :]
    else:
        new = img_rotated[int(diff[0]):-int(diff[0]),:]

    if diff[1] == 0.0:
        new = img_rotated[:, int(diff[1]):]
    else:
        new = img_rotated[:, int(diff[1]):-int(diff[1])]


    diff = (np.array(new.shape) - np.array(destination_shape))
    if diff[0] > 0:
        new = new[:-diff[0], :]
    
    if diff[1] > 0:
        new = new[:, :-diff[1]]

    new = new.astype(np.uint8)
    return new

def thresholding(img):
    thresh = filters.threshold_otsu(img, 3)
    img = img > thresh
    return img

def detect_lines_of_text(img):    
    # obliczamy średnią jasność wierszy w obrazie
    img = thresholding(img)
    img = morphology.erosion(img, morphology.disk(5))


    sum_of_rows = np.sum(img, axis=1) 
    mean_row_value = np.mean(sum_of_rows)

    max_value_in_row = img.shape[1]*1
    
    # wiersze poniżej średniej jasności zerujemy (usuwa np. ogonki liter nachodzących na kolejne wiersze)
    # chodzi o to żeby wyciągnąć na pewno te wiersze obrazu, w których jest tekst
    # zaznaczamy obszary, które są powyżej średniej (całe wiersze)e
    for i, row in enumerate(img):
        if np.sum(row) <= 0.02*max_value_in_row:
            row = np.zeros(row.shape)
            img[i] = row
        else:
            img[i:i+1, :] = 1       
            

    # łączenie lekko rozdzielonych linijek
#     img = morphology.dilation(img, morphology.disk(5)) # TODO WAŻNE - zoptymalizować!
                
    return img
            
def get_regions_of_rows(img_detected_rows, img_name='test'):
    # region = linia tekstu
    label_image = measure.label(img_detected_rows)
    regions = measure.regionprops(label_image)
    width = img_detected_rows.shape[1]

    # wybieramy wiersze powyżej 5 pikseli wysokości
    # regions = [reg for reg in regions if reg.area > width*5]

    # wybieramy regiony, które mają pole powierzchni większe od odchylenia standardowego
    std = np.std([reg.area for reg in regions])/width 

    big_regions = [reg for reg in regions if reg.area/img_detected_rows.shape[1] > std*0.75]
    small_regions = [reg for reg in regions if reg.area/img_detected_rows.shape[1] <= std*0.75]

    img_detected_rows = util.img_as_ubyte(img_detected_rows)
    for reg in small_regions:
        closest_reg = find_closest_region(all_regions=big_regions, region=reg)
        
        closest_reg_height = np.max(closest_reg.coords[:, 0])
        reg_height = np.max(reg.coords[:, 0])

        img_detected_rows[min(reg_height, closest_reg_height):max(reg_height, closest_reg_height), :] = np.iinfo(img_detected_rows.dtype).max
    
    label_image = measure.label(img_detected_rows)
    regions = measure.regionprops(label_image)

    ######################### TESTOWE #########################
    save_path = Path('data/partial_results/3/1_wykryte_wiersze_tekstu')
    save_path.mkdir(parents=True, exist_ok=True)
    io.imsave(arr=util.img_as_ubyte(img_detected_rows), fname=save_path / '{}_2.png'.format(img_name))
    ######################### TESTOWE #########################

    return regions


def find_closest_region(all_regions, region):
    """
        Wyszukuje najbliższy region względem osi Y (wysokość obrazu).
        
    """
    region_y_min = np.min(region.coords[:, 0])
    region_y_max = np.max(region.coords[:, 0])

    temp_distance = np.inf
    closest_region = None
    for reg in all_regions:
        current_reg_y_min = np.min(reg.coords[:, 0])
        current_reg_y_max = np.max(reg.coords[:, 0])
        
        is_reg_higher = True
        if current_reg_y_min > region_y_min:
            is_reg_higher = False
        
        if is_reg_higher:
            diff = region_y_min - current_reg_y_max
        else:
            diff = current_reg_y_min - region_y_max

        if diff < temp_distance:
            temp_distance = diff
            closest_region = reg
    
    return closest_region


def detect_words_in_line(image_result, image_binary, coords_of_line, reference_point_to_img_raw, row_intensity=255):
    # wycinamy kawałek obrazu będącego linią tekstu i obracamy go (.T)
    line_img = get_slice_of_image_with_specific_coords(image=image_binary, coords=coords_of_line).T
    line_img = thresholding(line_img)
    # print(line_img.shape)
    line_img = morphology.dilation(line_img, morphology.disk(13))
    
    # szukamy miejsc, w których jasność jest większa od 0.0 i te miejsca zaznaczamy w wycinku obrazu
    # (obraz jest obrócony, więc tekst idzie z góry na dół)
    sum_of_rows = np.sum(line_img, axis=1) 
    mean_row_value = np.mean(sum_of_rows)
    line_img = util.img_as_ubyte(line_img)
    for i, row in enumerate(line_img):
        if np.mean(row) > 0.0:
#             line_img[i:i+1, :] = (255, 0, 0)
            line_img[i:i+1, :] = row_intensity
    # znowu obracamy, tekst biegnie od lewej do prawej
    line_img = line_img.T

    # wykrywamy regiony, czyli pojedynczy region to powinien być jeden wyraz
    label_line_img = measure.label(line_img)
    regions = measure.regionprops(label_line_img)
#     print("liczba słów: ", len(regions))

    # tutaj szukamy regionu, który jest najdalej na lewo - czyli indeksu
    max_width_coord = max(regions[0].coords[:, 1])
    max_region_index = 0
    for i, region in enumerate(regions[1:]):        
        temp = max(region.coords[:, 1])
        if temp > max_width_coord:
            max_width_coord = temp
            max_region_index = i + 1

    # czyli mamy wszystkie współrzędne regionu z indeksem
    last_word_coords = regions[max_region_index].coords
    # ale aktualnie do tego regionu odnosimy się względem naszego wycinka obrazu - jednego wiersza
    # a chcemy go zaznaczyć na całym obrazie, więc do współrzędnych dodajemy współrzędne naszego wycinka,
    # (te współrzędne wycinka odnoszą się do całego obrazu)
    last_word_coords[:,0] += coords_of_line[0][0]

    # zamieniamy ten wycinek obrazu w całym obrazie
    first_point = coords_of_line[0]
    last_point = coords_of_line[-1]
    # image_result[first_point[0]+reference_point_to_img_raw[0]:last_point[0]+reference_point_to_img_raw[0]+1, first_point[1]+reference_point_to_img_raw[1]:last_point[1]+reference_point_to_img_raw[1]+1] = line_img
    image_result[first_point[0]:last_point[0]+1, first_point[1]:last_point[1]+1] = line_img
    return last_word_coords, image_result


def get_slice_of_image_with_specific_coords(image, coords):
    height = coords[:, 0]
    width = coords[:, 1]
    slice_image = image[(height, width)].reshape((-1, image.shape[1]))
    
    return slice_image


def detect_fragments_with_words(img, img_raw, gray_fragment, rotation, reference_point_to_img_raw, img_out_path_words, img_name='test'):
    """
    Zapisuje k-wyrazy.png oraz wykrywa fragmenty obrazu reprezentującego indeksy.
    
    Parameters:
    img: Na wejście dostaje obraz binarny, gdzie wykryty tekst jest wskazywany przez 1, a reszta to 0.
    img_raw: Oryginalny niezmieniony obraz.
    reference_point_to_img_raw: Punkt odniesienia wycinku (img) do obrazu oryginalnego (img_raw).
    
    Returns:
    last_word_images: Lista wyciętych fragmentów indeksów z oryginalnego obrazu.
    
    """    
    # Wyrównujemy tekst, aby był jak najbardziej poziomo.
    raw_img_shape = img.shape
    img = transform.rotate(img, rotation, resize=True, preserve_range=True)
    gray_fragment = transform.rotate(gray_fragment, rotation, resize=True, preserve_range=True)

    img_detected_rows = detect_lines_of_text(util.img_as_float(img.copy()))
    ######################### TESTOWE #########################
    save_path = Path('data/partial_results/3/1_wykryte_wiersze_tekstu')
    save_path.mkdir(parents=True, exist_ok=True)
    io.imsave(arr=util.img_as_ubyte(img_detected_rows), fname=save_path / '{}_1.png'.format(img_name))
    ######################### TESTOWE #########################

    print('1')
    regions = get_regions_of_rows(img_detected_rows, img_name=img_name)
    print('2')

    
    # Wynikowy obraz ma mieć czarne tło, a wyrazy w kolejnych wierszach mają mieć wartości 1,2,3...
    image_result_fragment = np.zeros(gray_fragment.shape, dtype=np.uint8)
    last_words = []
    for i, region in enumerate(regions, 1):
        last_word_coords, image_result_fragment = detect_words_in_line(image_result=image_result_fragment, 
                                                            image_binary=img, 
                                                            coords_of_line=region.coords, 
                                                            row_intensity=((i*1)%256),
                                                            reference_point_to_img_raw=reference_point_to_img_raw)
        
        
        # last_word_coords = np.array([[el[0]+reference_point_to_img_raw[0],el[1]+reference_point_to_img_raw[1]] for el in last_word_coords])
        last_words.append(last_word_coords)
    

    # Przekształcamy z powrotem na pochylony obraz jak oryginalnie
    image_result_fragment = transform.rotate(image_result_fragment, -rotation, preserve_range=True, resize=True)
    image_result_fragment = resize_after_rotate(img_rotated=image_result_fragment, destination_shape=raw_img_shape)

    # Tworzymy obraz wynikowy majacy wymiary identyczne jak oryginalny.
    image_result = np.zeros(img_raw.shape, dtype=np.uint8)
    height = reference_point_to_img_raw[0]
    width = reference_point_to_img_raw[1]
    # Wklejamy do niego fragment, na którym pracowaliśmy i zaznaczyliśmy wykryte słowa.
    image_result[height:height+image_result_fragment.shape[0], width:width+image_result_fragment.shape[1]] = image_result_fragment

    # Zapisywanie k-wyrazy 
    # image_result = util.img_as_ubyte(image_result)
    io.imsave(arr=image_result, fname=img_out_path_words)
    
    # Wycięcie indeksu (last_word) z oryginalnego obrazu i dodanie go do listy wszystkich.
    last_word_images = []
    margin = 7
    for i, last_word_coords in enumerate(last_words):
        first_point = last_word_coords[0]
        last_point = last_word_coords[-1]
        
        last_word_img = gray_fragment[max(first_point[0]-margin, 0):last_point[0]+margin, first_point[1]:last_point[1]+1] 
        last_word_images.append(last_word_img)

    ######################### TESTOWE #########################
    save_path = Path('data/partial_results/3/2_wyciete_indeksy/{}'.format(img_name))
    save_path.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(last_word_images):  
        img = util.img_as_ubyte(img)
        io.imsave(arr=img, fname=save_path / '{}.png'.format(i))
    ######################### TESTOWE #########################
    return last_word_images