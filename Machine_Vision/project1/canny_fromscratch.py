import torch
import torch.nn as nn
import sys
import math
import matplotlib.pyplot as plt
def pltshow(rx,gx,bx,ry,gy,by):
    rgbimgX=np.stack([rx.squeeze(),gx.squeeze(),bx.squeeze()],2)
    rgbimgY=np.stack([ry.squeeze(),gy.squeeze(),by.squeeze()],2)
    rgbimgX[rgbimgX<0]=0
    rgbimgY[rgbimgY<0]=0
    rgbimgX[rgbimgX>255]=255
    rgbimgY[rgbimgY>255]=255
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8))
    ax1.imshow(cv2.cvtColor(rgbimgX.astype('uint8'), cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(rgbimgY.astype('uint8'), cv2.COLOR_BGR2RGB))
    ax1.set_title('sobelX')
    ax2.set_title('sobelY')
    ax1.axis('off')
    ax2.axis('off')
    plt.savefig("/mnt/ve_share/liushuai/Document-Boundary-Detection/liushuai_2020212267/sobel_operator.jpg", bbox_inches='tight')
def show(r,g,b,path):
    toshow=np.stack([r.squeeze(),g.squeeze(),b.squeeze()],2)
    cv2.imwrite(path,toshow)
def initiate(filter_size=5, std=1.0, map_func=lambda x:x):
    kernel = torch.exp(-torch.arange(- filter_size// 2 + 1., filter_size// 2 + 1.) ** 2 / (2 * std ** 2))
    kernel2d=torch.ger(kernel,kernel).unsqueeze(0).unsqueeze(0).numpy().astype(np.float32)
    kernel2d/=np.sum(kernel2d)
    sobel_filter_horizontal = np.array([[[
        [1., 0., -1.], 
        [2., 0., -2.],
        [1., 0., -1.]]]], 
        dtype='float32'
    )

    sobel_filter_vertical = np.array([[[
        [1., 2., 1.], 
        [0., 0., 0.], 
        [-1., -2., -1.]]]], 
        dtype='float32'
    )

    directional_filter = np.array(
        [[[[ 0.,  0.,  0.],
          [ 0.,  1., -1.],
          [ 0.,  0.,  0.]]],


        [[[ 0.,  0.,  0.],
          [ 0.,  1.,  0.],
          [ 0.,  0., -1.]]],


        [[[ 0.,  0.,  0.],
          [ 0.,  1.,  0.],
          [ 0., -1.,  0.]]],


        [[[ 0.,  0.,  0.],
          [ 0.,  1.,  0.],
          [-1.,  0.,  0.]]],


        [[[ 0.,  0.,  0.],
          [-1.,  1.,  0.],
          [ 0.,  0.,  0.]]],


        [[[-1.,  0.,  0.],
          [ 0.,  1.,  0.],
          [ 0.,  0.,  0.]]],


        [[[ 0., -1.,  0.],
          [ 0.,  1.,  0.],
          [ 0.,  0.,  0.]]],


        [[[ 0.,  0., -1.],
          [ 0.,  1.,  0.],
          [ 0.,  0.,  0.]]]], 
        dtype=np.float32
    )

    connect_filter = np.array([[[
        [1., 1., 1.], 
        [1., 0., 1.], 
        [1., 1., 1.]]]],
        dtype=np.float32
    )

    return {
        'gaussian.weight': map_func(kernel2d),
        'sobel_filter_horizontal.weight': map_func(sobel_filter_horizontal),
        'sobel_filter_vertical.weight': map_func(sobel_filter_vertical),
        'directional_filter.weight': map_func(directional_filter),
        'connect_filter.weight': map_func(connect_filter)
    }
class CannyDetector(nn.Module):
    def __init__(self, filter_size=5, std=1.0, device='cpu'):
        super(CannyDetector, self).__init__()
        self.device = device
        # gaussian模糊
        self.gaussian = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,filter_size), padding=(filter_size//2,filter_size//2), bias=False)
        # Sobel滤波
        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        # 梯度方向滤波
        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, bias=False)
        # 中心点模糊滤波
        self.connect_filter = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        # initiate
        params = initiate(filter_size=filter_size, std=std, map_func=lambda x:torch.from_numpy(x).to(self.device))
        self.load_state_dict(params)

    @torch.no_grad()
    def forward(self, img, lowvalue=10.0, highvalue=100.0):
        # 拆分图像通道
        img_r = img[:,:,0] # red channel
        img_g = img[:,:,1] # green channel
        img_b = img[:,:,2] # blue channel
        tofloat=lambda x:torch.tensor(x).float().unsqueeze(0)
        
        # gaussianblured preprocess
        blurred_img_r = self.gaussian(tofloat(img_r))
        blurred_img_g = self.gaussian(tofloat(img_g))
        blurred_img_b = self.gaussian(tofloat(img_b))
        # blurred_img_r=blurred_img_r.squeeze()
        # blurred_img_g=blurred_img_g.squeeze()
        # blurred_img_b=blurred_img_b.squeeze()
        # to test the gaussianblured function 
        show(blurred_img_r,blurred_img_g,blurred_img_b,"/mnt/ve_share/liushuai/Document-Boundary-Detection/liushuai_2020212267/gauss_fromscratch.jpg")
        
        # Sobel operator
        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)
        pltshow(grad_x_r,grad_x_g,grad_x_b,grad_y_r,grad_y_g,grad_y_b)
        # calculate grad
        calgrad=lambda x,y:torch.sqrt(x**2+y**2) #the function of grad calculation
        grad_mag = calgrad(grad_x_r,grad_y_r)
        grad_mag += calgrad(grad_x_g,grad_x_g)
        grad_mag += calgrad(grad_x_b,grad_y_b)
        grad_orientation = (torch.atan2(grad_y_r+grad_y_g+grad_y_b, grad_x_r+grad_x_g+grad_x_b) * (180.0/math.pi))
        grad_orientation += 180.0
        grad_orientation =  torch.round(grad_orientation / 45.0) * 45.0

        all_filtered = self.directional_filter(grad_mag) #八个梯度方向的featuremap
        inidices_positive = (grad_orientation / 45) % 8 #梯度方向
        inidices_negative = ((grad_orientation / 45) + 4) % 8 #加上180度 进行反向索引
        _, height, width = inidices_positive.shape 
        num_pixel = height * width
        pixel_range = torch.Tensor([range(num_pixel)]).to(self.device)
        indices = (inidices_positive.reshape((-1, )) * num_pixel + pixel_range).squeeze()
        channel_select_filtered_positive = all_filtered.reshape((-1, ))[indices.long()].reshape((1, height, width))
        #实际在哪个梯度有效
        indices = (inidices_negative.reshape((-1, )) * num_pixel + pixel_range).squeeze()
        channel_select_filtered_negative = all_filtered.reshape((-1, ))[indices.long()].reshape((1, height, width))
        channel_select_filtered = torch.stack([channel_select_filtered_positive, channel_select_filtered_negative])
        is_max = channel_select_filtered.min(dim=0)[0] > 0.0 #找出两个向量中相同位置较小的那个值，并将其与0进行比较（如果为true说明中间比他邻域的相同方向都大）
        final_edge = grad_mag.clone()
        final_edge[is_max==0] = 0.0
        
        #double threshold
        low = min(lowvalue, highvalue)
        high = max(lowvalue, highvalue)
        thresholded = final_edge.clone()
        lower = final_edge<low
        thresholded[lower] = 0.0 #小于较小阈值的设为0
        higher = final_edge>high
        thresholded[higher] = 1.0#大于较大阈值的设为1
        connect_map = self.connect_filter(higher.float())#对大于阈值的部分进行模糊处理
        middle = torch.logical_and(final_edge>=low, final_edge<=high)#居中的
        thresholded[middle] = 0.0 #把居中的设为0
        connect_map[torch.logical_not(middle)] = 0
        thresholded[connect_map>0] = 1.0
        thresholded[..., 0, :] = 0.0
        thresholded[..., -1, :] = 0.0
        thresholded[..., :, 0] = 0.0
        thresholded[..., :, -1] = 0.0 #去除四个边缘
        thresholded = (thresholded>0.0).float()

        return thresholded,grad_mag


class Hough_transform:
    def __init__(self, img, grad_mag, mindistance, step, circle_threshold):
        self.img = img
        self.grad_mag = grad_mag.squeeze(0)
        self.height, self.width = img.shape[0:2]
        self.radius = math.ceil(math.sqrt(self.height**2 + self.width**2))   #圆直径的最大长度
        self.mindistance = mindistance
        self.step = step
        self.vote_matrix = np.zeros([math.ceil(self.height / self.step), math.ceil(self.width / self.step), math.ceil(self.radius / self.step)])
        self.circle_threshold = circle_threshold
        self.circles = []

    def Hough_transform_algorithm(self):
        for height in range(1, self.height - 1):
            for width in range(1, self.width - 1):
                if self.img[height][width] > 0: #像素值大于0，则作为圆心，以步长为r扩散半径
                    x = width
                    y = height
                    r = 0
                    while x > 0 and y > 0 and x < self.width and y < self.height:
                        self.vote_matrix[math.floor(y / self.step)][math.floor(x / self.step)][math.floor(r / self.step)] += 1
                        x = x + self.step
                        y = y + self.grad_mag[height][width] * self.step
                        r = r + math.sqrt(self.step ** 2 + (self.grad_mag[height][width] * self.step) ** 2)
                    x = width - self.step
                    y = height - self.grad_mag[height][width] * self.step
                    r = math.sqrt(self.step ** 2 + (self.grad_mag[height][width] * self.step) ** 2)
        return self.vote_matrix


    def Select_Circle(self):        
        candidateCircles = []
        for i in range(0, self.vote_matrix.shape[0]):
            for j in range(0, self.vote_matrix.shape[1]):
                for k in range(0, self.vote_matrix.shape[2]):
                    if self.vote_matrix[i][j][k] > self.circle_threshold:
                        y = i * self.step + (self.step / 2)
                        x = j * self.step + (self.step / 2)
                        r = k * self.step + (self.step / 2)
                        candidateCircles.append([math.ceil(x), math.ceil(y), math.ceil(r)])

        x, y, r = candidateCircles[0]
        possibleCircles = []
        middleCircles = []
        for circle in candidateCircles:
            if math.sqrt((x - circle[0])**2 + (y - circle[1])**2) <= self.mindistance: #如果两个圆的距离小于阈值,则可能是同一个圆的候选点
                possibleCircles.append([circle[0], circle[1], circle[2]])
            else:
                result = np.array(possibleCircles).mean(axis=0)#如果大于，将所有圆的平均位置计算出来     
                middleCircles.append([result[0], result[1], result[2]])
                possibleCircles.clear()
                x, y, r = circle
                possibleCircles.append([x, y, r])
        result = np.array(possibleCircles).mean(axis=0)
        middleCircles.append([result[0], result[1], result[2]])

        middleCircles.sort(key=lambda x:x[0], reverse=False)
        x, y, r = middleCircles[0]
        possibleCircles = []
        for circle in middleCircles:
            if math.sqrt((x - circle[0])**2 + (y - circle[1])**2) <= self.mindistance:
                possibleCircles.append([circle[0], circle[1], circle[2]])
            else:
                result = np.array(possibleCircles).mean(axis=0)
                print("Circle core: (%f, %f), Radius: %f" % (result[0], result[1], result[2]))
                self.circles.append([result[0], result[1], result[2]])
                possibleCircles.clear()
                x, y, r = circle
                possibleCircles.append([x, y, r])
        result = np.array(possibleCircles).mean(axis=0)
        self.circles.append([result[0], result[1], result[2]])

    def printcircle(self):
        for circle in self.circles:
            x,y,r=circle
            cv2.circle(img, (x.astype('uint8'), y.astype('uint8')), r.astype('uint8'), (255, 0, 0), 2)
        cv2.imwrite('/mnt/ve_share/liushuai/Document-Boundary-Detection/liushuai_2020212267/circle_fromscratch.jpg', img)
    def Calculate(self):
        self.Hough_transform_algorithm()
        self.Select_Circle()
        self.printcircle()
        return self.circles
if __name__ == '__main__':
    import cv2
    import numpy as np
    img = cv2.imread("/mnt/ve_share/liushuai/Document-Boundary-Detection/liushuai_2020212267/coin.jpg")
    canny_operator = CannyDetector()
    result,gradmag=canny_operator(img)
    res = np.squeeze(result.numpy())
    res = (res*255).astype(np.uint8)
    circle=Hough_transform(res,gradmag,150,100,30)
    circle.Calculate()
    cv2.imwrite("/mnt/ve_share/liushuai/Document-Boundary-Detection/liushuai_2020212267/edge_fromscratch.jpg",res)