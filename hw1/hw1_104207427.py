import numpy as np
import cv2
from scipy import signal
from scipy import misc


def convolve3x3(fter, img):
    img2 = np.copy(img)

    for row in range(img.shape[0]-2):
        for col in range(img.shape[1]-2):
            total = 0.0

            for i in range(3):
                for j in range(3):
                    if(fter.sum()!=0):
                        total = fter[i, j] * img[i + row, j + col]/fter.sum() + total
                    else:
                        total = fter[i, j] * img[i + row, j + col] + total
            img2[1 + row, 1 + col] = total
    return img2

def kernel(kernel3x3_name):
    file = open(kernel3x3_name)
    Mx = np.zeros([3, 3])
    row = 0


    for line in file.readlines():
        line = line.strip('\n')
        col = 0
        j = 0
        while j < len(line):
            if(line[j]=='-'):
                Mx[row, col] = float(line[j+1])*-1
                j+=2
                col+=1
            elif(line[j]==','):
                j+=1
            else:
                Mx[row, col] = line[j]
                col+=1
                j+=1
        row += 1
    return Mx

img = cv2.imread('lena.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

sharpen = convolve3x3(kernel('sharpen.csv'), img_gray)
cv2.imshow('sharpen',sharpen)

blur = convolve3x3(kernel('blur.csv'), img_gray)
cv2.imshow('blur',blur)

outline = convolve3x3(kernel('outline.csv'), img_gray)
cv2.imshow('outline',outline)

sharpen_f = signal.convolve2d(img_gray, kernel('sharpen.csv'))
cv2.imshow('sharpen_f',sharpen_f)


print(img.shape)
cv2.waitKey(0)
