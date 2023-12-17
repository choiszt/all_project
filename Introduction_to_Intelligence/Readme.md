## 模型功能

通过使用生成对抗网络，生成类似手写数字的图像

## 环境

python 3.7以上 GPU最好支持CUDA

Pytorch1.8.1以上. 

## 启动命令

在shell当中打开gan文件夹  [Windows/Liunx]命令无差异

输入

`python gan.py`

回车运行

## 异常处理

可能会因为文件路径的差异而导致缺少数据集,下载数据集比较缓慢

建议将数据集路径改为

`../data/mnist`

## 结果数据查看

实验结果在image文件夹中(与 gan.py 同一文件夹下)

`cd image`

即可查看
