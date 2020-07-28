from pathlib import Path
from skimage import filters
from skimage import transform
from skimage import measure
from skimage import exposure
from skimage import color
from skimage import util
from skimage import morphology
from matplotlib import pyplot as plt

import numpy as np

# TESTOWE
from skimage import io
from skimage import draw
from matplotlib import pyplot as plt

# rect_points - (start_point, end_point), where p0 is top-left corner, p1 is down-right corner
def euclidean_distance(p1, p2):
        return pow(pow(p1[0]-p2[0],2)+pow(p1[1]-p2[1],2),0.5)


def sort_regions_by_area(regions, descending=True):     
    def func(region):
        return region.area

    regions = sorted(regions, key=func, reverse=descending)
    return regions

def get_binary_image_with_digits(word_image):
    # Multio-Otsu dzieli obraz na 3 klasy o różnych jasnościach.
    thresholds = filters.threshold_multiotsu(word_image, classes=3)

    # Regions - to obraz, który ma wartości od 0 do 2. 
    # Wartość odpowiada regionowi, do którego należey dany piksel.
    otsu_regions = np.digitize(word_image, bins=thresholds)

    # Jeden z wykrytych regionów odpowiadał w większości jasności cyfr..
    # Region trzeba traktować jako jakiś przedział wartości jasności w obrazie.
    # image_removed_otsu_region = word_image*util.invert((otsu_regions == 1))
    image_digits = (otsu_regions==0)


    # Po ponownym wykryciu regionów, jeden z nich pasował do cyfr więc użyłem go licząc, że są to cyfry.
    # thresholds = filters.threshold_multiotsu(image_removed_otsu_region, classes=3)
    # otsu_regions = np.digitize(image_removed_otsu_region, bins=thresholds)
    # image_digits = (otsu_regions==0)
    
    # image_digits = morphology.dilation(image_digits, morphology.disk(1)) # DODANE
    

    return image_digits

def get_digits_regions(image_digits):
    # Tutaj region to już chodzi o podobne jasności pikseli w sąsiedztwie.
    label_image = measure.label(image_digits)
    regions = measure.regionprops(label_image)
    regions = [reg for reg in regions if reg.area > 50]
    
    # usuwanie regionów, które są zbyt szerokie na liczbę (usuwa wykryte poziome linie kratki)
    # źle działa dla 2_1, cyfra 5 jest zbyt duża
#     width = image_digits.shape[1] 
#     regions_width = [(np.max(reg.coords[:, 1]) - np.min(reg.coords[:, 1])) for reg in regions]
#     regions = [reg for i, reg in enumerate(regions) if regions_width[i] < width/5]

#     print("liczba wykrytych regionów (cyfr): ", len(regions))
    
    
    return regions


def remove_overlapped_regions(regions_as_rect_points):
    # usunięcie regionów zbytnio nakładajacych się na siebie, prawdopodobnie cyfra została podzielona na górę-dół
    
    regions_list_with_key = []
    for i, reg in enumerate(regions_as_rect_points):
        regions_list_with_key.append([i, reg])
    
    valid_regions = []
    invalid_regions_keys = []
    l1 = regions_list_with_key[0:]
    l2 = regions_list_with_key[1:]
    for reg1_dict, reg2_dict in zip(l1, l2): 
        reg1_key, reg1 = reg1_dict
        reg2_key, reg2 = reg2_dict
        
        if reg1_key in invalid_regions_keys:
            continue

        reg1_start_point, reg1_end_point = reg1
        reg2_start_point, reg2_end_point = reg2
        
        diff = reg2_start_point[1] - reg1_end_point[1] 
        if diff < 0: # jeżeli regiony oddalone o mniej niż 'diff' pikseli to połącz w jeden
            new_start_point, new_end_point = combine_two_overlapping_regions(reg1, reg2)
                
            valid_regions.append([new_start_point, new_end_point])              
            invalid_regions_keys.append(reg1_key)
            invalid_regions_keys.append(reg2_key)

    
    for key, reg in regions_list_with_key:
        if key not in invalid_regions_keys:
            valid_regions.append(reg)
            
    return valid_regions

def combine_two_overlapping_regions(reg1, reg2):
    reg1_start_point, reg1_end_point = reg1
    reg2_start_point, reg2_end_point = reg2
        
    new_start_point = []
    new_end_point = []

    if reg1_start_point[0] < reg2_start_point[0]:
        new_start_point.append(reg1_start_point[0])
    else:
        new_start_point.append(reg2_start_point[0])
    if reg1_start_point[1] < reg2_start_point[1]:
        new_start_point.append(reg1_start_point[1])
    else:
        new_start_point.append(reg2_start_point[1])           

    if reg1_end_point[0] > reg2_end_point[0]:
        new_end_point.append(reg1_end_point[0])
    else:
        new_end_point.append(reg2_end_point[0])
    if reg1_end_point[1] > reg2_end_point[1]:
        new_end_point.append(reg1_end_point[1])
    else:
        new_end_point.append(reg2_end_point[1])
        
    return new_start_point, new_end_point

def get_list_of_rectangle_points(regions):
    rect_points = [get_two_points_to_create_rectangle_from_region(reg) for reg in regions]
    
    # usuwanie bardzo małych wykrytych obszarów
    def func_area(points):
        start_point, end_point = points
        width = abs(end_point[1]-start_point[1])
        height = abs(end_point[0]-start_point[0])
        return width*height
    
    rect_points_area = [func_area(reg) for reg in rect_points]
    rect_points = [reg for i, reg in enumerate(rect_points) if rect_points_area[i] > 20]
    
    # sortowanie obszarów, biegnących od lewej do prawej - aby cyfry były po kolei
    def func(points):
        p0, p1 = points
        return (p0[1] + p1[1])/2   
    rect_points_sorted_by_distance_to_start_of_horizontal_axis = sorted(rect_points, key=func)    
    rect_points = remove_overlapped_regions(rect_points_sorted_by_distance_to_start_of_horizontal_axis)
    rect_points = sorted(rect_points, key=func) 
    rect_points = remove_overlapped_regions(rect_points)
    rect_points = sorted(rect_points, key=func) 
    
#     rect_points = rect_points_sorted_by_distance_to_start_of_horizontal_axis # TODO do usunięcia

#     l1 = rect_points[0:]
#     l2 = rect_points[1:]
#     for reg1, reg2 in zip(l1, l2):
#         half_diff = int((reg2[0][1] - reg1[1][1])/2)
#         reg1[1][1] += half_diff
#         reg2[0][1] -= half_diff
        
#     print("liczba wykrytych regionów ostatecznie (cyfr): ", len(rect_points))
#     print(rect_points)
    return rect_points
        

def get_two_points_to_create_rectangle_from_region(region):
    height_min = np.min(region.coords[:, 0])
    width_min = np.min(region.coords[:, 1])
    height_max = np.max(region.coords[:, 0])
    width_max = np.max(region.coords[:, 1])
    
    return [[height_min, width_min], [height_max, width_max]]

def scale_digit_image(image, scale=28):
    (h, w) = image.shape
    
    if abs(h-w) < 2:
        image = transform.resize(image, (scale,scale))
        return image
    if h > w: 
        half_diff = int((h-w)/2) # zakładamy, że jednak zawsze cyfra jest wyższa niż szersza
        
        # left border
        new_image_left_border = np.full((h,w+half_diff), fill_value=0, dtype=np.uint8)
        new_image_left_border[:,half_diff:] = image
        image = new_image_left_border
              
        # right border
        fill_value = (h-w) - 2*half_diff
        (h, w) = image.shape
        new_image_right_border = np.full((h,w+half_diff+fill_value), fill_value=0, dtype=np.uint8)
        new_image_right_border[:,:-half_diff-fill_value] = image
        image = new_image_right_border
        
    else:
        half_diff = int((w-h)/2) # zakładamy, że jednak zawsze cyfra jest wyższa niż szersza

        # bottom border  
        new_image_bot_border = np.full((h+half_diff,w), fill_value=0, dtype=np.uint8)
        new_image_bot_border[:-half_diff,:] = image
        image = new_image_bot_border

        # top border
        fill_value = (w-h) - 2*half_diff
        (h, w) = image.shape
        new_image_top_border = np.full((h+half_diff+fill_value,w), fill_value=0, dtype=np.uint8)
        new_image_top_border[half_diff+fill_value:,:] = image
        image = new_image_top_border
    
    image = transform.resize(image, (scale,scale)) # to zmienia na float8
    image = util.img_as_bool(image) # jak tego nie użyłem to po zmianie na uint8 (czyli akcja niżej) 
                                    # miałem wartości inne niż 0 i 255 (np. 254)
    image = util.img_as_ubyte(image) # nie mogę zapisywać obrazów typu bool, czyli takich jak wyżej, więc zmiana na uint8
    
    return image


def cut_digits_from_index_image(last_word_images, img_name='test'):
    """
    Wycina cyfry z obrazu indeksu i zapisuje wykryte indeksy do k-indeksy.txt.
    
    Parameters:
    last_word_images: List obrazów indeksów.
    
    """   

    ######################### TESTOWE #########################
    save_path_img = Path('data/partial_results/4/1_wyciete_cyfry/{}'.format(img_name))
    save_path_img.mkdir(parents=True, exist_ok=True)
    ######################### TESTOWE #########################
    
    all_indexes_list = []
    for i, word_image_org in enumerate(last_word_images):
        # Wczytanie obrazu
        index_digits_list = []
        word_image = word_image_org.copy()
        # word_image = color.rgb2gray(word_image)
        word_image = filters.gaussian(word_image)


        # Rozciąganie jasności obrazu.
        p2, p98 = np.percentile(word_image, (2, 98))
        word_image = exposure.rescale_intensity(word_image, in_range=(p2, p98))

        image_digits = get_binary_image_with_digits(word_image)

        regions = get_digits_regions(image_digits)  

        rect_points_sorted_by_distance_to_start_of_horizontal_axis = get_list_of_rectangle_points(regions)

        word_image = color.gray2rgb(word_image)  
        image_digits = util.img_as_ubyte(image_digits)
        

        ######################### TESTOWE #########################
        save_path_word = save_path_img / str(i)
        save_path_word.mkdir(parents=True, exist_ok=True)

        temp_image = word_image_org.copy()
        temp_image = color.gray2rgb(temp_image)
        ######################### TESTOWE #########################

        for index_digit, (start_point, end_point) in enumerate(rect_points_sorted_by_distance_to_start_of_horizontal_axis):           
            # Wycięcie cyfry
            one_digit =  image_digits[start_point[0]:end_point[0]+1, start_point[1]:end_point[1]+1]
            # one_digit =  image_digits[:, start_point[1]:end_point[1]+1]

            one_digit = scale_digit_image(one_digit, scale=28)
            index_digits_list.append(one_digit)


            ######################### TESTOWE #########################
            io.imsave(arr=one_digit, fname=save_path_word / '{}.png'.format(index_digit))

            # Narysowanie prostokąta wokół cyfry.
            rr, cc = draw.rectangle_perimeter(start_point, end_point, shape=temp_image.shape)         
            temp_image[rr, cc] = (255,0+index_digit*30,0+index_digit*30)
            io.imsave(arr=temp_image, fname=save_path_word / 'index.png')
            ######################### TESTOWE #########################
         
        all_indexes_list.append(index_digits_list)


    return all_indexes_list