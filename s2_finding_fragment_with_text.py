from skimage import measure,exposure,transform
import numpy as np


# TESTOWE
from pathlib import Path
from skimage import io
from skimage import util


def remove_outliers_centroids(data, quantile_height=0.9, quantile_width=0.9):
    Q1 = np.quantile(data[:, 0], 1-quantile_height)
    Q3 = np.quantile(data[:, 0], quantile_height)
    IQR = Q3 - Q1
    data = np.array([el for el in data if el[0] > Q1 and el[0] < Q3])

    Q1 = np.quantile(data[:, 1], 1-quantile_width)
    Q3 = np.quantile(data[:, 1], quantile_width)
    IQR = Q3 - Q1
    data = np.array([el for el in data if el[1] > Q1 and el[1] < Q3])
    
    return data

def remove_background_on_left_side(img):
    img = util.img_as_float(img)
    
    ############ Jeżeli zwracać obraz w skali szarości, bez rozciągania histogramu to użyć tego ############
    ############ ,a w głównej funkcji usunąć ############
#     p2, p98 = np.percentile(img, (2, 98))
#     img = exposure.rescale_intensity(img, in_range=(p2, p98))
    ###########################################################################################
    
    # Sprawdzamy średnią jasność obrazu. Następnie bierzemy po kolei grupy kolumn (skłądające się z step kolumn).
    # Jeżeli jasność w grupie jest mniejsza niż (średnia w obrazie - bias) to zastępujemy wartością średnią.
    mean_value_in_img = np.mean(img) 
    step = 10
    bias = 0.2 # 0.07 bez roziągania histogramu
    was_action = True 

    for el in range(0, len(img.T), step):
        if np.mean(img[:,el:el+step]) < mean_value_in_img-bias:
#             img[:,el:el+step] = mean_value_in_img
            img = img[:,el+step:]
            was_action = True
        else:
            was_action = False
        
        if not was_action:
#             img[:,el:el+step] = mean_value_in_img
            img = img[:,el+int(step/2):]
            break
            
    return img

def remove_background(img):
    # usuwa z lewej
    img = remove_background_on_left_side(img)
    
    # usuwa z prawej
    img = transform.rotate(img, 180)
    img = remove_background_on_left_side(img)

    # usuwa z dołu
    img = img.T
    img = remove_background_on_left_side(img)

    # usuwa z góry
    img = transform.rotate(img, 180)
    img = remove_background_on_left_side(img)

    # wraca do oryginalnej postaci
    img = img.T
    
    return img


def detect_fragment_with_text(img, img_raw, img_name="test"):
    """
    Parameters:
    img: Na wejście dostaje obraz binarny, gdzie wykryte tekst jest wskazywany przez 1, a reszta to 0.
    img_raw: Oryginalny niezmieniony obraz.
    
    Returns:
    img_removed_background: Zwraca wycinek obrazu zawierający tekst w skali szarości z rozciągniętym histogramem.
    
    """
    
    # Szukamy regionów. Więkoszość z nich powinna znajdować się w obszarze tekstu.
    label_img = measure.label(img)
    regions = measure.regionprops(label_img)
    regions_centroids = np.array([reg.centroid for reg in regions])
    mean_centroid = (np.mean(regions_centroids[:, 0]), np.mean(regions_centroids[:, 1]))

    # Usuwamy obszary, który centoridy zbyt mocno odstają. Dwukrotnie.
    data = remove_outliers_centroids(regions_centroids, quantile_height=0.95, quantile_width=0.9)
    data = remove_outliers_centroids(data, quantile_height=0.95, quantile_width=0.95)

    # Z pozostałych centoridów tworzymy prostokąt troche powiększony.
    height_min = np.min(data[:, 0])
    width_min = np.min(data[:, 1])
    height_max = np.max(data[:, 0])
    width_max = np.max(data[:, 1])
    
    img_height, img_width = img.shape
    
    start_point_height = int(max(height_min-img_height*0.09, 1))
    start_point_width = int(max(width_min-img_width*0.15, 1))
    end_point_height = int(height_max+img_height*0.1)
    end_point_width = int(width_max+img_width*0.25)
    
    # Wycięcie tego prostokątu z oryginalnego obrazu.
#     img_raw_path = Path('../data/ocr1') / str(image_path.stem + ".jpg")
#     img_raw = io.imread(img_raw_path)
    img_slice = img_raw[start_point_height:end_point_height, start_point_width:end_point_width]
    
    # Na wycinkach nadal czasami pojawia się stół. Więc usuwamy te fragmenty.
    p2, p98 = np.percentile(img_slice, (2, 98))
    img_slice = exposure.rescale_intensity(img_slice, in_range=(p2, p98))

    shape_before = np.array(img_slice.shape)
    img_removed_background = remove_background(img_slice)
    shape_after = np.array(img_removed_background.shape)
    shape_diff = (shape_before-shape_after)

    reference_point_to_img_raw = np.array([start_point_height, start_point_width]) + shape_diff

    ######################### TESTOWE #########################
    save_path = Path('data/partial_results/2_wyciete_fragmenty')
    save_path.mkdir(parents=True, exist_ok=True)
    io.imsave(arr=util.img_as_ubyte(img_removed_background), fname=save_path / (img_name+'.png'))
    ######################### TESTOWE #########################


    return img_removed_background, reference_point_to_img_raw
