import cv2 
from matplotlib import pyplot as plt
from pathlib import Path
from skimage import filters,exposure,img_as_ubyte
import numpy as np
import sys
import skimage
from skimage.filters import threshold_otsu
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage import data
from matplotlib import cm
from skimage.transform import probabilistic_hough_line,rotate
from skimage import util
# TESTOWE
from skimage import io




def get_words_from_base_img(img):
    
    kernel = np.ones((8,8),np.uint8)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
#     plt.gcf().set_size_inches(30, 20)
#     plt.imshow(gradient,cmap = 'gray'),plt.title('gradient')
#     plt.show() 

    kernel = np.ones((7,7),np.uint8)
    erosion = cv2.erode(gradient,kernel,iterations = 1)
    
    kernel = np.ones((3,3),np.uint8)
    closing1 = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)
    
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(closing1,kernel,iterations = 2)
    
    
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(erosion,kernel,iterations = 1)
    
    kernel = np.ones((3,3),np.uint8)
    dilation2 = cv2.dilate(dilation,kernel,iterations = 3)
    
    
    erosion2 = cv2.bitwise_not(dilation2)
    
    # plt.gcf().set_size_inches(30, 20)
    # plt.imshow(erosion2,cmap = 'gray'),plt.title('erosion2')
    # plt.show() 
    
    kernel = np.ones((3,3),np.uint8)

    closing2 = cv2.morphologyEx(erosion2, cv2.MORPH_CLOSE, kernel)
    closing3 = cv2.morphologyEx(closing2, cv2.MORPH_CLOSE, kernel)
#     closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
    # plt.gcf().set_size_inches(30, 20)
    # plt.imshow(closing3,cmap = 'gray'),plt.title('closing')
    # plt.show() 
    

    edges = cv2.Canny(closing2,50,100)
#     plt.gcf().set_size_inches(30, 20)
#     plt.imshow(edges,cmap = 'gray'),plt.title('edges')
# #     plt.savefig('22_better.jpg')
#     plt.show()
    return edges

def trh(img):
    global_thresh = threshold_otsu(img)
    binary_global = img > global_thresh
    return binary_global 

def sobel_get_img_from_background(img, img_name='test'):  
    img = util.img_as_ubyte(img)  
    rotation = hugh_and_rotation(img)
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=7)
    sobely = cv2.Sobel(sobelx,cv2.CV_64F,0,1,ksize=7)
#     plt.gcf().set_size_inches(30, 20)
#     plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
#     plt.title('Original'), plt.xticks([]), plt.yticks([])
#     plt.subplot(1,2,1),plt.imshow(laplacian,cmap = 'gray')
#     plt.subplot(1,2,2),plt.imshow(sobely,cmap = 'gray')
#     plt.title('Sobel Y+X'), plt.xticks([]), plt.yticks([])
#     plt.show()
    
    kernelg = np.ones((5,5),np.uint8)

    gradient = cv2.morphologyEx(sobely, cv2.MORPH_GRADIENT, kernelg)
    
    kernelgo =  np.ones((5,5),np.uint8)
    opening1 = cv2.morphologyEx(gradient, cv2.MORPH_OPEN, kernelgo,iterations = 2)
    
    kernele = np.ones((3,3),np.uint8)
    erosion = cv2.erode(opening1,kernele,iterations = 1)
    
    kernelgo2 =  np.ones((3,3),np.uint8)
    opening2 = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernelgo2,iterations = 2)
    
    # plt.gcf().set_size_inches(11, 7)
    # plt.imshow(opening2,cmap = 'gray'),plt.title('opening2')

    plt.show()
    imgg = skimage.filters.gaussian(opening2) 
    opening3 = cv2.morphologyEx(imgg, cv2.MORPH_OPEN, kernelgo2,iterations = 4)
    imgg2 = skimage.filters.gaussian(opening3) 
    
    new_img = trh(imgg2)
    # plt.gcf().set_size_inches(11, 7)
    # plt.imshow(new_img,cmap = 'gray'),plt.title('normalizedImg')

    new_image = img_as_ubyte(new_img)
    
    ######################### TESTOWE #########################
    save_path = Path('data/partial_results/1/2_kontury_wyrazow_na_fragmencie')
    save_path.mkdir(parents=True, exist_ok=True)
    io.imsave(arr=new_image, fname=save_path / (img_name+'.png'))
    ######################### TESTOWE #########################

    return new_image,rotation



#WYKRYWANIE KĄTA NA RAZIE NIE UŻYWANE!!!!!!!

def get_angle(line):
    angle = np.rad2deg(np.arctan2(line[1][1] - line[0][1], line[1][0] - line[0][0]))
    return angle
    
def hugh_and_rotation(image):
    kernelg = np.ones((3,3),np.uint8)
    gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernelg)
    edges = cv2.Canny(gradient,100,100)
    angle  = 0
    # plt.gcf().set_size_inches(11, 7)
    # plt.imshow(gradient,cmap = 'gray'),plt.title('closing1')
    # plt.show()
    # plt.gcf().set_size_inches(11, 7)
    # plt.imshow(edges,cmap = 'gray'),plt.title('closing1')
    # plt.show() 
    
    
    angles = np.linspace((np.pi / 2)-0.040, (np.pi / 2)+0.040, 360)
    lines = probabilistic_hough_line(edges, threshold=10, line_length=600,
                                     line_gap=25,theta = angles)

# #     Generating figure 2
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
#     ax = axes.ravel()

#     ax[0].imshow(image, cmap=cm.gray)
#     ax[0].set_title('Input image')

#     ax[1].imshow(edges, cmap=cm.gray)
#     ax[1].set_title('Canny edges')

#     ax[2].imshow(edges * 0)
#     for line in lines:
#         p0, p1 = line
#         ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
#     ax[2].set_xlim((0, image.shape[1]))
#     ax[2].set_ylim((image.shape[0], 0))
#     ax[2].set_title('Probabilistic Hough')
    
    if len(lines) > 0:
        angle = get_angle(lines[0])
        image = rotate(image, angle)
        # print(angle)

    
    return angle
    
    
# img = cv2.imread(OBRAZY+'/img_20.png',0)


# image2, lines = hugh_and_rotation(img)

# plt.gcf().set_size_inches(11, 7)
# plt.imshow(img,cmap = 'gray'),plt.title('Before rotation')
# plt.show() 

# plt.gcf().set_size_inches(11, 7)
# plt.imshow(image2,cmap = 'gray'),plt.title('After rotation')
# plt.show() 