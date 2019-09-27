import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def drawHist(src, fig_name):

    h = np.shape(src)[0]
    w = np.shape(src)[1]
    arr = np.zeros([256], dtype = np.int32)
    for row in range(h):
        for col in range(w):
            pv = src[row,col]
            arr[pv] +=1 #calculate num of pixel value

    plot_name = fig_name
    plt.figure()
    plt.title(plot_name)
    plt.plot(arr,color = "b")
    plt.xlim([0,255])
    plt.show()


def equalization(src):
    h = np.shape(src)[0]
    w = np.shape(src)[1]
    arr = np.zeros([256], dtype=np.int32)
    for row in range(h):
        for col in range(w):
            pv = src[row, col]
            arr[pv] += 1
    cdf = np.zeros([256], dtype=np.int32)
    cdf[0] = arr[0]
    for num_arr in range(254):
        cdf[num_arr+1] = arr[num_arr+1] + cdf[num_arr]
    hv = np.zeros([256], dtype = np.float)
    cdfmin = cdf[0]
    for hv_value in range(255):
        hv[hv_value] = round((cdf[hv_value]-cdfmin)/(h*w-cdfmin)*(255),0)

    for row in range(h):
        for col in range(w):
            src[row,col]=hv[src[row, col]]

    return src


def RGB(mp2a):
# Q2a: mp2a's RGB channel equalization
    mp2a_clone =mp2a.copy()

    ht = np.shape(mp2a_clone)[0]
    wd = np.shape(mp2a_clone)[1]
    mp2a_B = np.zeros((ht, wd),dtype = mp2a.dtype)
    mp2a_G = np.zeros((ht, wd),dtype = mp2a.dtype)
    mp2a_R = np.zeros((ht, wd),dtype = mp2a.dtype)
    E_mp2a_B = np.zeros((ht, wd),dtype = mp2a.dtype)
    E_mp2a_G = np.zeros((ht, wd),dtype = mp2a.dtype)
    E_mp2a_R = np.zeros((ht, wd),dtype = mp2a.dtype)
    mp2a_merge_np = np.zeros((ht, wd),dtype = mp2a.dtype)

    mp2a_B = mp2a_clone[:,:,0]
    mp2a_G = mp2a_clone[:,:,1]
    mp2a_R = mp2a_clone[:,:,2]
    drawHist(mp2a_clone[:, :, 0], "mp2a_B")
    drawHist(mp2a_clone[:, :, 1], "mp2a_G")
    drawHist(mp2a_clone[:, :, 2], "mp2a_R")

    E_mp2a_B = equalization(mp2a_B)
    E_mp2a_R = equalization(mp2a_R)
    E_mp2a_G = equalization(mp2a_G)

    mp2a_merge_cv = cv.merge([E_mp2a_B, E_mp2a_G, E_mp2a_R]) #try the merge funtion in cv2
    mp2a_merge_np = np.dstack([E_mp2a_B, E_mp2a_G, E_mp2a_R]) #try the merge function in numpy
    cv.imshow("mp2a_RGB_equalization", mp2a_merge_np)

    drawHist(E_mp2a_G,"Equal_G")
    drawHist(E_mp2a_R,"Equal_R")
    drawHist(E_mp2a_B,"Equal_B")

    #equalizeHist function in cv
    eq_B = cv.equalizeHist(mp2a_B)
    eq_G = cv.equalizeHist(mp2a_G)
    eq_R = cv.equalizeHist(mp2a_R)
    eq_merge = cv.merge([eq_B, eq_G, eq_R])
    cv.imshow("CV_func_RGB_equalizeHist()", eq_merge )

def HSV(mp2a):
# Q2b: mp2a's RGB convert to HSV, V channel's equalization

    mp2a_clone2b = mp2a.copy()
    ht = np.shape(mp2a_clone2b)[0]
    wd = np.shape(mp2a_clone2b)[1]
    mp2a_hsv = cv.cvtColor(mp2a_clone2b, cv.COLOR_BGR2HSV)
    mp2a_HSV_clone = mp2a_hsv.copy()
    mp2a_hsv_v = np.zeros((ht,wd),dtype = mp2a.dtype)
    mp2a_hsv_v = mp2a_HSV_clone[:,:,2]
    drawHist(mp2a_hsv_v, "hsv")
    E_hsv_v = equalization(mp2a_hsv_v)
    drawHist(E_hsv_v, "V-Equal")
    hsv_merge_np = np.dstack([mp2a_HSV_clone[:,:,0], mp2a_HSV_clone[:,:,1], E_hsv_v])
    cv.imshow("HSV_equalization_hsv", hsv_merge_np)
    hsv2bgr = cv.cvtColor(hsv_merge_np, cv.COLOR_HSV2BGR)
    cv.imshow("HSV_equalization_rgb", hsv2bgr)

    func_hsv_v = mp2a_HSV_clone[:,:,2]
    eq_v = cv.equalizeHist(func_hsv_v)
    func_hsv_merge = np.dstack([mp2a_HSV_clone[:,:,0], mp2a_HSV_clone[:,:,1], eq_v])
    eq_merge = cv.cvtColor(func_hsv_merge, cv.COLOR_HSV2BGR)
    cv.imshow("CV_func_HSV_equalizeHist()", eq_merge)

def YCbCr(mp2a):
# Q2c: mp2a's RGB convert to YCbCr, Y channel's equalization

    mp2a_clone2c = mp2a.copy()
    ht = np.shape(mp2a_clone2c)[0]
    wd = np.shape(mp2a_clone2c)[1]
    mp2a_YCrCb = cv.cvtColor(mp2a_clone2c, cv.COLOR_BGR2YCrCb)
    mp2a_YCrCb_clone = mp2a_YCrCb.copy()
    mp2a_YCrCb_y = np.zeros((ht,wd),dtype = mp2a.dtype)
    mp2a_YCrCb_y = mp2a_YCrCb_clone[:,:,0]
    drawHist(mp2a_YCrCb_y, "YCrCb")
    E_YCrCb_y = equalization(mp2a_YCrCb_y)
    drawHist(E_YCrCb_y, "Y-Equal")
    YCrCb_merge_np = np.dstack([E_YCrCb_y, mp2a_YCrCb_clone[:,:,1],mp2a_YCrCb_clone[:,:,2] ])
    cv.imshow("YCbCr_equalization_YCbCr", YCrCb_merge_np)
    YCrCb2bgr = cv.cvtColor(YCrCb_merge_np, cv.COLOR_YCrCb2BGR)
    cv.imshow("YCbCr_equalization_RGB", YCrCb2bgr)

    func_YCrCb_Y = mp2a_YCrCb_clone[:,:,0]
    eq_Y = cv.equalizeHist(func_YCrCb_Y)
    func_YCrCb_merge = cv.merge([eq_Y, mp2a_YCrCb_clone[:,:,1],mp2a_YCrCb_clone[:,:,2]])
    eq_merge = cv.cvtColor(func_YCrCb_merge, cv.COLOR_YCrCb2BGR)
    cv.imshow("CV_func_YCbCr_equalizeHist()", eq_merge)

if __name__ == '__main__':
    mp2 = cv.imread('mp2.jpg')
    mp2_clone = mp2.copy()


    # Q1: mp2's equalization
    drawHist(mp2_clone, "mp2")
    drawHist(equalization(mp2_clone), "equal_mp2")
    cv.imshow("mp2-equalization", mp2_clone)

    mp2a = cv.imread('mp2a.jpg')
    RGB(mp2a)
    HSV(mp2a)
    YCbCr(mp2a)

    if(cv.waitKey(0) == 27):
        cv.destroyAllWindows()