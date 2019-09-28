import cv2
import numpy as np
import matplotlib.pyplot as plt

img_1 = cv2.imread('kobe_freethrow.jpg',0)#('img1.png',0)
img_2 = cv2.imread('mj_freethrow.jpg',0)#('img2.png',0)

cv2.imshow('origin1', img_1)
cv2.imshow('origin2', img_2)
f1 = np.fft.fft2(img_1)
f1shift = np.fft.fftshift(f1)
f1_A = np.abs(f1shift)
f1_P = np.angle(f1shift)
f2 = np.fft.fft2(img_2)
f2shift = np.fft.fftshift(f2)
f2_A = np.abs(f2shift)
f2_P = np.angle(f2shift)
#---圖1的振幅--圖2的相位--------------------
img_new1_f = np.zeros(img_1.shape,dtype=complex)
img1_real = f1_A*np.cos(f2_P) #取實部
img1_imag = f1_A*np.sin(f2_P) #取虚部
img_new1_f.real = np.array(img1_real)
img_new1_f.imag = np.array(img1_imag)
f3shift = np.fft.ifftshift(img_new1_f) #對新的進行逆變換
img_new1 = np.fft.ifft2(f3shift)
#出來的是複數 無法顯示
img_new1 = np.abs(img_new1)
#調整大小範圍便於顯示
img_new1 = (img_new1-np.amin(img_new1))/(np.amax(img_new1)-np.amin(img_new1))

cv2.imshow('img1_amp & img2_phase', img_new1)

#---圖1的相位--圖2的振幅--------------------
img_new2_f = np.zeros(img_1.shape,dtype=complex)
img2_real = f2_A*np.cos(f1_P)
img2_imag = f2_A*np.sin(f1_P)
img_new2_f.real = np.array(img2_real)
img_new2_f.imag = np.array(img2_imag)
f4shift = np.fft.ifftshift(img_new2_f)
img_new2 = np.fft.ifft2(f4shift)
img_new2 = np.abs(img_new2)
img_new2 = (img_new2-np.amin(img_new2))/(np.amax(img_new2)-np.amin(img_new2))
cv2.imshow('img1_phase & img2_amp', img_new2)

'''
#---圖1,2的振幅相加--圖1,2的相位相加--------------------
img_new3_f = np.zeros(img_1.shape,dtype=complex)
img3_real = (f1_A+f2_A)*np.cos(f2_P+f1_P)
img3_imag = (f1_A+f2_A)*np.sin(f2_P+f1_P)
img_new3_f.real = np.array(img3_real)
img_new3_f.imag = np.array(img3_imag)
f5shift = np.fft.ifftshift(img_new3_f)
img_new3 = np.fft.ifft2(f5shift)
#出來的是複數 無法顯示
img_new3 = np.abs(img_new3)
#調整大小範圍便於顯示
img_new3 = (img_new3-np.amin(img_new3))/(np.amax(img_new3)-np.amin(img_new3))

cv2.imshow('img1,2_amp & img1,2_phase', img_new3)
'''
cv2.waitKey(0)