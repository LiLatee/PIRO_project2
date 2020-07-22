import cv2 
OBRAZY = '/home/adam/studia/piro/proj2/PIRO_project2/data/ocr1'
img = cv2.imread(OBRAZY+'/img_1.jpg')
cv2.imshow("test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()