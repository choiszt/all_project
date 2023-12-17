import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
train_loader=torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data',train=True,download=False,
                               transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.11),(0.3))])),
                               batch_size=100,shuffle=True)
test_loader=torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data',train=False,download=False,
                               transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.13,),(0.31))])),
                               batch_size=10000,shuffle=True)

class NN(nn.Module):
    def __init__(self,input_size,num_classes):
        super(NN,self).__init__()
        self.fc1=nn.Linear(input_size,50)
        self.fc2=nn.Linear(50,num_classes)
        self.dp=nn.Dropout(0.5)

    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        x=self.dp(x)
        x=self.fc2(x)
        return x
device=torch.device('cuda'if torch.cuda.is_available()else 'cpu')
in_channel=1
input_size=784
num_classes=10
learning_rate=0.001
batch_size=64
num_epochs=2
model=NN(input_size,num_classes).to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(num_epochs):
    print('epoch=',epoch)
    loss=[]
    model.train()
    for data,label in tqdm(train_loader):
        data=data.to(device=device)
        data = data.reshape(data.shape[0], -1)
        label=label.to(device=device)
        pred=model(data)
        loss=criterion(pred,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

num_correct=0
num_samples=0
model.eval()
with torch.no_grad():
    for data,label in tqdm(test_loader):
        data=data.to(device=device)
        label=label.to(device=device)
        data=data.reshape(data.shape[0],-1)
        print(model(data).argmax(dim=1)==label)
        num_correct+=(model(data).argmax(dim=1)==label).sum()
        print(model(data).shape)
        num_samples+=model(data).size(0)
print('correctness:',num_correct)
print('sampleness:',num_samples)
print('accuracy:',num_correct/num_samples)