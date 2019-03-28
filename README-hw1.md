1. 利用cv2.imread() function 讀入圖片
2. 將其藉由cv2.cvtColor(img, COLOR_RGB2GRAY) function 
把圖片從RGB轉換為 grayscale picture.
3. 撰寫 convolve3x3(fter, img ) 的function 
fter 需放入未正規化的矩陣
Img 需放入要被卷積的影像

在function中，先複製影像，逐一取出各pixel值並乘上filter中設定的權重，
最後加總放回中心的位置

4. 撰寫讀取kernel(kernel3x3_name) 的function
kernel3x3_name 須為3*3的矩陣 並且檔案格式為csv檔

5.最後將不同kernel放入lena.jpg中測試效果
利用 convolve3x3(fter, img) function 實作不同
Kernel造成的卷積效果
在這邊我試用了 sharpen, outline和blur三種
