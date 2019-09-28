import cv2
import numpy as np


def harris_detect(img, threshold, ksize=3):

    k = 0.04
    nms= False


    h, w = img.shape[:2]
    grad = np.zeros((h, w, 2), dtype=np.float32)
    grad[:, :, 0] = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    grad[:, :, 1] = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)

    # calculate Ix^2,Iy^2,Ix*Iy
    m = np.zeros((h, w, 3), dtype=np.float32)
    m[:, :, 0] = grad[:, :, 0] ** 2
    m[:, :, 1] = grad[:, :, 1] ** 2
    m[:, :, 2] = grad[:, :, 0] * grad[:, :, 1]

    # gaussian filter Ix^2,Iy^2,Ix*Iy
    m[:, :, 0] = cv2.GaussianBlur(m[:, :, 0], ksize=(ksize, ksize), sigmaX=2)
    m[:, :, 1] = cv2.GaussianBlur(m[:, :, 1], ksize=(ksize, ksize), sigmaX=2)
    m[:, :, 2] = cv2.GaussianBlur(m[:, :, 2], ksize=(ksize, ksize), sigmaX=2)
    m = [np.array([[m[i, j, 0], m[i, j, 2]], [m[i, j, 2], m[i, j, 1]]]) for i in range(h) for j in range(w)]

    # calculate D=det(M), T=trace(M)
    # calculate each pixel, R(i,j)=det(M)-k(trace(M))^2  0.04<=k<=0.06
    D, T = list(map(np.linalg.det, m)), list(map(np.trace, m))
    R = np.array([d - k * t ** 2 for d, t in zip(D, T)])

    # 對R值進行非極大值壓制，filter掉不是corner的點，&& 滿足>threshold
    # 獲取最大的R值
    R_max = np.max(R)
    # print(R_max)
    # print(np.min(R))
    R = R.reshape(h, w)
    corner = np.zeros_like(R, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if nms:
                if (R[i, j] > R_max * threshold) and (R[i, j] == np.max(R[max(0, i-1):min(i+2, h-1), max(0, j-1):min(j+2, w-1)])):
                    corner[i, j] = 255
            else:

                if (R[i, j] > R_max * threshold):
                    corner[i, j] = 255
    return corner


if __name__ == '__main__':
    input_file = 'example2.jpeg'

    img = cv2.imread(input_file)
    img = cv2.resize(img, dsize=(600, 400))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dst = harris_detect(gray, 0.0001)
    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imshow('corner', img)
    cv2.waitKey(0)
