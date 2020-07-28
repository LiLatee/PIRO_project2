import numpy as np
from pathlib import Path
from skimage import io
from skimage import measure
from skimage import morphology
from matplotlib import pyplot as plt
from skimage import util
# %matplotlib qt

def detect_lines_of_text(img):    
    # obliczamy średnią jasność wierszy w obrazie
    sum_of_rows = np.sum(img, axis=1) 
    mean_row_value = np.mean(sum_of_rows)
    
    # wiersze poniżej średniej jasności zerujemy (usuwa np. ogonki liter nachodzących na kolejne wiersze)
    # chodzi o to żeby wyciągnąć na pewno te wiersze obrazu, w których jest tekst
    for i, row in enumerate(img):
        if np.sum(row) < mean_row_value: # mean_row_value, ale to jest takie niefajne, eh
            row = np.zeros(row.shape)
            img[i] = row
        
    # zaznaczamy obszary, które są powyżej średniej (całe wiersze)
    for i, row in enumerate(img):
        if np.sum(row) > 0.0:
            img[i:i+1, :] = 1 
            
    # łączenie lekko rozdzielonych linijek
#     img = morphology.dilation(img, morphology.disk(5)) # TODO WAŻNE - zoptymalizować!
                
    return img
            

def detect_words_in_line(image_result, image_binary, coords_of_line, row_intensity=255):
    # wycinamy kawałek obrazu będącego linią tekstu i obracamy go (.T)
    line_img = get_slice_of_image_with_specific_coords(image=image_binary, coords=coords_of_line).T
    line_img = morphology.dilation(line_img, morphology.disk(13))
    
    # szukamy miejsc, w których jasność jest większa od 0.0 i te miejsca zaznaczamy w wycinku obrazu
    # (obraz jest obrócony, więc tekst idzie z góry na dół)
    sum_of_rows = np.sum(line_img, axis=1) 
    mean_row_value = np.mean(sum_of_rows)
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
    image_result[first_point[0]:last_point[0]+1, first_point[1]:last_point[1]+1] = line_img
                
    return last_word_coords


def get_slice_of_image_with_specific_coords(image, coords):
    height = coords[:, 0]
    width = coords[:, 1]
    slice_image = image[(height, width)].reshape((-1, image.shape[1]))
    
    return slice_image


save_path_k_wyrazy = Path('../data/partial_results/k_wyrazy_2') # POTRZEBNE - to jeden z wyników zadania
save_path_k_wyrazy.mkdir(parents=True, exist_ok=True)

def detect_fragments_with_words(img, img_org, reference_point_to_img_org,img_out_path_words):
    """
    Zapisuje k-wyrazy.png oraz wykrywa fragmenty obrazu reprezentującego indeksy.
    
    Parameters:
    img: Na wejście dostaje obraz binarny, gdzie wykryty tekst jest wskazywany przez 1, a reszta to 0.
    img_org: Oryginalny niezmieniony obraz.
    reference_point_to_img_org: Punkt odniesienia wycinku (img) do obrazu oryginalnego (img_org).
    
    Returns:
    last_word_images: Lista wyciętych fragmentów indeksów z oryginalnego obrazu.
    
    """    
    img_detected_rows = detect_lines_of_text(img.copy()) 

    # region = linia tekstu
    label_image = measure.label(img_detected_rows)
    regions = measure.regionprops(label_image)
    
#     width = img_canny.shape[1]
#     regions = [reg for reg in regions if reg.area > width*5] # wiersze powyżej 7 pikseli wysokości
#     print("regions po usunięciu cienkich wierszy: ", len(regions))
    
    # Wynikowy obraz ma mieć czarne tło, a wyrazy w kolejnych wierszach mają mieć wartości 1,2,3...
    image_result = np.zeros(img.shape, dtype=np.uint8)
    last_words = []
    for i, region in enumerate(regions, 1):
        last_word_coords = detect_words_in_line(image_result=image_result, 
                                               image_binary=img, 
                                               coords_of_line=region.coords, 
                                               row_intensity=((i*1)%256))
        
        last_word_coords_height = last_word_coords[0] + reference_point_to_img_org[0]
        last_word_coords_width = last_word_coords[1] + reference_point_to_img_org[1]
        last_words.append([last_word_coords_height, last_word_coords_width])
    
    # Zapisywanie k-wyrazy 
    # image_result = util.img_as_ubyte(image_result)
    io.imsave(arr=image_result, fname=img_out_path_words)

    
    # Utworzenie katalogu dla wycinka indeksu.
#     number_of_image = re.search('[0-9]+', image_path.stem)[0]
#     last_word_directory = save_path / number_of_image
#     last_word_directory.mkdir(parents=True, exist_ok=True)
    
    # Wycięcie indeksu (last_word) z oryginalnego obrazu i dodanie go do listy wszystkich.
    last_word_images = []
    print(last_words)
    for i, last_word_coords in enumerate(last_words):
        first_point = last_word_coords[0]
        last_point = last_word_coords[-1]
        last_word_img = img_org[first_point[0]:last_point[0]+1, first_point[1]:last_point[1]+1] 
        print(last_word_img)
        last_word_images.append(last_word_img)
        # Zapisanie
#         io.imsave(arr=last_word_img, fname=last_word_directory / '{}.png'.format(i))

    return last_word_images