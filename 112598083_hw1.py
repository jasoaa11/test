import numpy as np
import cv2


def grayscale(img):
    return  np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

def convolution(img,kernel):
    padded_array=np.zeros((img.shape[0] + 2, img.shape[1] + 2))
    result = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            padded_array[i+1][j+1]=img[i][j]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_array[i:i+3, j:j+3]
            result[i, j] = np.sum(region * kernel)
    return result

def maxPool(img):
    height = img.shape[0] // 2
    width = img.shape[1] // 2
    result = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            window = img[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2]
            result[i, j] = np.max(window)
    return result

def binarization(img,threshold):
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if(img[x ,y] < threshold):
                img[x ,y] = 0
            else:
                img[x, y] = 255
    return img


#Q1
img = cv2.imread("test_img/aeroplane.png")
grayscale_img = grayscale(img)
#cv2.imwrite("result_img/aeroplane_Q1.png",grayscale_img)  
img2 = cv2.imread("test_img/taipei101.png")
grayscale_img2 = grayscale(img2)
#cv2.imwrite("result_img/taipei101_Q1.png",grayscale_img2)  


#Q2
kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
convolution_img=convolution(grayscale_img,kernel)
#cv2.imwrite("result_img/aeroplane_Q2.png",convolution_img)  
convolution_img2=convolution(grayscale_img2,kernel)
#cv2.imwrite("result_img/taipei101_Q2.png",convolution_img2) 

#Q3
maxpool_img=maxPool(convolution_img)
#cv2.imwrite("result_img/aeroplane_Q3.png",maxpool_img)
maxpool_img2=maxPool(convolution_img2)
#cv2.imwrite("result_img/taipei101_Q3.png",maxpool_img2)

#Q4
binarization_img=binarization(maxpool_img,128)
#cv2.imwrite("result_img/aeroplane_Q4.png",binarization_img)
binarization_img2=binarization(maxpool_img2,128)
#cv2.imwrite("result_img/taipei101_Q4.png",binarization_img2)

