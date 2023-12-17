import numpy as np
import cv2
import matplotlib.pyplot as plt
def GaussianBlur(sigma,img):
    sigma = float(sigma)
    num1 = np.around( (2 * np.pi * sigma ** 2) ** (-1),decimals=7)
    num2 = np.around(  (2 * np.pi * sigma ** 2) ** (-1) * np.exp((np.negative(sigma ** 2)) ** (-1) * 0.5),decimals=7)
    num3 = np.around( (2 * np.pi * sigma ** 2) ** (-1) * np.exp((np.negative(sigma ** 2)) ** (-1)),decimals=7)
    GaussMatrix = np.array([[num3,num2,num3],
                            [num2,num1,num2],
                            [num3,num2,num3]])
    total = np.around( ( (num2+num3)*4 + num1),decimals=7)   
    img = cv2.copyMakeBorder(img,1,1,1,1,borderType=cv2.BORDER_REPLICATE)
    (b,g,r) = cv2.split(img)
    b1 = np.zeros(b.shape,dtype="uint8")
    g1 = np.zeros(g.shape,dtype="uint8")
    r1 = np.zeros(r.shape,dtype="uint8")
    temp = list(range(3))
    for i in range(1,b.shape[0]-1):
        for j in range(1,b.shape[1]-1):
            temp[0] = int( (np.dot(np.array([1, 1, 1]), GaussMatrix * b[i - 1:i + 2, j - 1:j + 2]/total)).dot(np.array([[1], [1], [1]])))
            temp[1] = int((np.dot(np.array([1, 1, 1]),  GaussMatrix * g[i - 1:i + 2, j - 1:j + 2]/total)).dot(np.array([[1], [1], [1]])))
            temp[2] = int((np.dot(np.array([1, 1, 1]),  GaussMatrix * r[i - 1:i + 2, j - 1:j + 2]/total)).dot(np.array([[1], [1], [1]])))

            b1[i, j] = temp[0]
            g1[i, j] = temp[1]
            r1[i, j] = temp[2]
    b1=b1[1:-1,1:-1]
    g1=g1[1:-1,1:-1]
    r1=r1[1:-1,1:-1] #把图像还原为原始size
    image = cv2.merge([b1,g1,r1])
    return image

dir="/mnt/ve_share/liushuai/Document-Boundary-Detection/liushuai_2020212267/coin.jpg"
image = cv2.imread(dir)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8))
ax1.imshow(cv2.cvtColor(cv2.GaussianBlur(image, (3,3), 0), cv2.COLOR_BGR2RGB))
ax2.imshow(cv2.cvtColor(GaussianBlur(100,image), cv2.COLOR_BGR2RGB))
ax1.set_title('offical3x3GaussianBlur')
ax2.set_title('my3x3gaussianblur')
ax1.axis('off')
ax2.axis('off')
plt.savefig("/mnt/ve_share/liushuai/Document-Boundary-Detection/liushuai_2020212267/gaussblur.jpg", bbox_inches='tight')
image=cv2.GaussianBlur(image, (5, 5), 0)#高斯模糊处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #转化为灰度图
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY) #二值化处理，将大于200的像素设为255，反之设为0
edged = cv2.Canny(gray, 75, 200) #利用canny算子进行边缘提取，设置低阈值为75，高阈值为200
cv2.imwrite("/mnt/ve_share/liushuai/Document-Boundary-Detection/liushuai_2020212267/edge_opencv.jpg",edged)
circles = cv2.HoughCircles(edged,cv2.HOUGH_GRADIENT ,0.1,120,param1=10,param2=30,minRadius=20,maxRadius=100)
if circles is not None:
    circles = np.uint16(np.around(circles))
    i=1
    for x,y,r in circles[0]:
        cv2.circle(image,(x,y),r,(255,0,0),3)
        print(f"第{i}个圆的中心坐标为({x},{y}),半径为:{r}")
        i+=1        
cv2.imwrite("/mnt/ve_share/liushuai/Document-Boundary-Detection/liushuai_2020212267/circle_opencv.jpg",image)