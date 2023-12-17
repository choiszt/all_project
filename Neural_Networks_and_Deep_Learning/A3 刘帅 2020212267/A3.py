import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.models as model
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

writer=SummaryWriter()
device=torch.device('cuda'if torch.cuda.is_available()else 'cpu')

train_dataset=torchvision.datasets.MNIST('../data',train=True,download=False,
                               transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.11),(0.3))]))
test_dataset=torchvision.datasets.MNIST('../data',train=False,download=False,
                               transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.13,),(0.31))]))

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, num_workers=4)

class CNN(nn.Module):
    def __init__(self,in_channels=1,num_classes=10):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(3,3),stride=(1,1),padding=(1,1))#same convolution
        self.pool=nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))#14*14
        self.conv2=nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.BN2d=nn.BatchNorm2d(16)
        self.fc1=nn.Linear(16*7*7,4096)
        self.dp=nn.Dropout(0.5)
        self.BN=nn.BatchNorm1d(4096)
        self.fc2=nn.Linear(4096,num_classes)
    def forward(self,x,tag=None):
        x=F.relu(self.conv1(x))
        x=self.pool(x)
        x=F.relu(self.conv2(x))
        x=self.BN2d(x)
        x=self.pool(x)
        x=x.reshape(x.shape[0],-1)#64*784
        x=self.fc1(x)
        if(tag=='withDPBN'):
            x=self.dp(x)
            x=self.BN(F.relu(x))
        elif(tag=='withDP'):
            x=self.dp(x)
        elif(tag=='withBN'):
            x=self.BN(F.relu(x))
        x=self.fc2(x)
        return x
cfgs = {
    'A': [64, 'M', 128,  256, 256, 'M', 512, 512,  512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
class VGG(nn.Module):

    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x,cfg=None):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs).to(device)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
def vgg_bn(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg_bn', 'A', True, pretrained, progress, **kwargs)

best_accuracy=0
def test_eval():
    global best_accuracy
    num_correct=0
    num_samples=0
    mymodel.eval()
    with torch.no_grad():
        for data,label in tqdm(test_loader):
            data=data.to(device=device)
            label=label.to(device=device)
            # data=data.reshape(data.shape[0],-1)
            # print(model(data).argmax(dim=1)==label)
            num_correct+=(mymodel(data).argmax(dim=1)==label).sum()
            # print(model(data).shape)
            num_samples+=mymodel(data).size(0)
    print('accuracy:',num_correct/num_samples)
    if num_correct/num_samples>best_accuracy:
        best_accuracy=num_correct / num_samples
    torch.save(mymodel,'CNNmodel.pth')
    return num_correct/num_samples

config=0
modelconfig=['withDPBN','withDP','withBN',None]
in_channel=1
input_size=784
num_classes=10
learning_rate=0.001
batch_size=64
num_epochs=5
kfold=KFold(n_splits=4,shuffle=True)

#mymodel=torch.nn.DataParallel(CNN(),device_ids=[0,1,2,3]).to(device)
mymodel=torch.nn.DataParallel(vgg_bn(),device_ids=[0,1,2,3])


criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(mymodel.parameters(),lr=learning_rate)
step = 0
for fold,(train_ids,val_ids) in enumerate(kfold.split(train_dataset)):
    train_subsampler=torch.utils.data.SubsetRandomSampler(train_ids)
    val_subsampler=torch.utils.data.SubsetRandomSampler(val_ids)
    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=100,sampler=train_subsampler,num_workers=4)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, sampler=val_subsampler, num_workers=4)


    train_data_sample,_ = iter(train_loader).next()
    with writer:
        writer.add_graph(mymodel.module, train_data_sample.to(device))

    for epoch in range(num_epochs):
        loss=[]
        num_correct=0
        num_samples=0
        train_losses=0
        mymodel.train()
        for data,label in tqdm(train_loader,desc="epoch={}".format(epoch)):
            data=data.to(device=device)
            label=label.to(device=device)
            pred=mymodel(data,modelconfig[config])
            loss=criterion(pred,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            with torch.no_grad():
                data=data.to(device=device)
                label=label.to(device=device)
                pred=mymodel(data,modelconfig[config])
                num_correct += (pred.argmax(dim=1) == label).sum()
                num_samples += pred.size(0)
                train_losses += loss.item()
        accuracy = num_correct / num_samples
        train_losses/=len(train_loader)
        print(f'fold:{fold},epoch:{epoch} Train - Loss:{train_losses} Accuracy:{accuracy}')
        step+=1
        writer.add_scalars('trainloss',{'kfold':fold,'epoch':epoch,'trainloss':train_losses},step)
        writer.add_scalars('trainacc', {'kfold': fold, 'epoch': epoch, 'trainacc': accuracy}, step)
        writer.add_text('trainlog',f'fold:{fold},epoch:{epoch} Train - Loss:{train_losses} Accuracy:{accuracy}',step)
        writer.flush()
        writer.close()

        num_correct=0
        num_samples=0
        val_losses=0
        with torch.no_grad():
            for data,label in tqdm(val_loader):
                data=data.to(device=device)
                label=label.to(device=device)
                pred=mymodel(data,modelconfig[config])
                loss=criterion(pred,label)
                num_correct+=(pred.argmax(dim=1)==label).sum()
                num_samples+=pred.size(0)
                val_losses+=loss.item()
        accuracy=num_correct/num_samples
        print(f'fold:{fold},epoch:{epoch} Val - Loss:{loss} Accuracy:{accuracy}')
        writer.add_scalars('valloss',{'kfold':fold,'epoch':epoch,'trainloss':train_losses},step)
        writer.add_scalars('valacc', {'kfold': fold, 'epoch': epoch, 'trainacc': accuracy}, step)
        writer.add_text('testlog',f'fold:{fold},epoch:{epoch} Val - Loss:{loss} Accuracy:{accuracy}',step)
        writer.flush()
        writer.close()

acc=test_eval()
writer.add_scalar("loss",loss,epoch)
writer.add_scalar("acc",acc,epoch)
writer.add_scalars('test_accuracy', acc)
writer.flush()
writer.close()

