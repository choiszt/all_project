import numpy as np
import struct


def loadImageSet(filename):
    binfile = open(filename, 'rb')  # 读取二进制文件
    buffers = binfile.read()

    head = struct.unpack_from('>IIII', buffers, 0)  # 取前4个整数，返回一个元组

    offset = struct.calcsize('>IIII')  # 定位到data开始的位置
    imgNum = head[1]
    width = head[2]
    height = head[3]

    bits = imgNum * width * height  # data一共有60000*28*28个像素值
    bitsString = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'

    imgs = struct.unpack_from(bitsString, buffers, offset)  # 取data数据，返回一个元组

    binfile.close()
    imgs = np.reshape(imgs, [imgNum, width * height])  # reshape为[60000,784]型数组

    return imgs, head


def loadLabelSet(filename):
    binfile = open(filename, 'rb')  # 读二进制文件
    buffers = binfile.read()

    head = struct.unpack_from('>II', buffers, 0)  # 取label文件前2个整形数

    labelNum = head[1]
    offset = struct.calcsize('>II')  # 定位到label数据开始的位置

    numString = '>' + str(labelNum) + "B"  # fmt格式：'>60000B'
    labels = struct.unpack_from(numString, buffers, offset)  # 取label数据

    binfile.close()
    labels = np.reshape(labels, [labelNum])  # 转型为列表(一维数组)

    return labels, head


if __name__ == "__main__":
    file1 = './data/train-images.idx3-ubyte'
    file2 = './data/train-labels.idx1-ubyte'
    file3='./data/t10k-images.idx3-ubyte'
    file4='./data/t10k-labels.idx1-ubyte'


    imgs, data_head = loadImageSet(file1)
    labels, labels_head = loadLabelSet(file2)
    np.save("../data/train_data.npy", imgs)
    np.save("../data/train_label.npy", labels)

    imgs, data_head = loadImageSet(file3)
    labels, labels_head = loadLabelSet(file4)
    np.save("../data/test_data.npy", imgs)
    np.save("../data/test_label.npy", labels)