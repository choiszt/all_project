import argparse
import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch

os.makedirs("images", exist_ok=True)

#定义超参数
parser = argparse.ArgumentParser()
parser.add_argument("--nums_of_epochs", type=int, default=500, help="训练轮数")
parser.add_argument("--batch_size", type=int, default=64, help="batch大小")
parser.add_argument("--lr", type=float, default=0.0002, help="优化器lr")
parser.add_argument("--latent_dim", type=int, default=100, help="隐藏层维度")
parser.add_argument("--img_size", type=int, default=28, help="图片大小")
parser.add_argument("--channels", type=int, default=1, help="通道数")
parser.add_argument("--sample_interval", type=int, default=1000, help="采样间隔")
parser.add_argument("--b1", type=float, default=0.5, help="一阶矩估计的指数衰减率")
parser.add_argument("--b2", type=float, default=0.999, help="二阶矩估计的指数衰减率")
opt = parser.parse_args()
print(opt)
img_shape = (opt.channels, opt.img_size, opt.img_size)
cuda = True if torch.cuda.is_available() else False
#利用nn.Module继承generator，采用5层感知机实现
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            *block(1024, 2048),
            *block(2048, 4096),
            nn.Linear(4096, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img

#定义discriminator 采用三层感知机模型
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),#每一层随机失活一半的神经元，增加模型的泛化程度
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity#返回一个0，1标签

two_elements_crossentropy = torch.nn.BCELoss()#二分类交叉熵损失函数 此处也可以用NLLLOSS，效果比BCELOSS稍差

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator() 

if cuda:#调用cuda，利用gpu处理
    generator.cuda()
    discriminator.cuda()
    two_elements_crossentropy.cuda()

#建立数据集
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(#在torchvision中调用下载调用mnist数据集，并建立dataloader
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,#batch内随机打乱
)

#利用Adam作为优化器（曾使用过sgd作为optimizer，但效果不好，最后出来的图片全为‘1’，怀疑是陷入局部最优）
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

#训练过程
for epoch in range(opt.nums_of_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)#定义全1向量，代表真实值
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)#定义全0向量，代表虚假值

        real_imgs = Variable(imgs.type(Tensor))


        optimizer_G.zero_grad()

        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))#输入高斯噪声

        gen_imgs = generator(z)#由噪声生成的虚假图片


        g_loss = two_elements_crossentropy(discriminator(gen_imgs), valid)#判定虚假图片和真实标签的交叉熵

        g_loss.backward()
        optimizer_G.step()



        optimizer_D.zero_grad()

        #求d的损失函数
        real_loss = two_elements_crossentropy(discriminator(real_imgs), valid)
        fake_loss = two_elements_crossentropy(discriminator(gen_imgs.detach()), fake)
        #在计算d_loss时只需要将fake_img.detach() 将fake_img与计算图“脱钩” 就不会影响d_loss.backward(
        # )的计算，因为更新只需要【fake_img -> D】这一段计算图。
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        #训练过程可视化
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.nums_of_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
