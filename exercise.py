# coding = utf-8
from unittest import TestLoader
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from models import *
import argparse
from utils import progress_bar
import os

# parameter definement
BATCH_SIZE = 200
EPOCHS = 100

# super parameters
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr',default=0.01,type=float,help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

# preprocess of dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# download data
training_data = datasets.CIFAR10(
    root = "/home/cjm/fourier/autoencoder/data",
    train = True,
    download = False,
    transform = transform
)

test_data = datasets.CIFAR10(
    root = "/home/cjm/fourier/autoencoder/data",
    train = False,
    download = False,
    transform = transform
)

# create data loaders
trainloader = DataLoader(training_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=16,pin_memory=True)
testloader = DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=16,pin_memory=True)

#print dataset
# for X,y in testloader:
#     print(X.shape)
#     print(y.shape)

#get device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# model
net = VGG('VGG16') # 注：最后一层是全连接层，输入该层的node数与输入图像大小有关; net输出[batch,10]
net = net.to(device)
# print(net)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr=args.lr)

# train
def train(dataloader,net,loss_fn,optimizer):
    net.train()
    training_loss = 0
    rate = 0
    for batch,(X,y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)

        #compute prediction error
        pred = net(X)
        loss = loss_fn(pred,y)
        training_loss += loss.item()

        #backprogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = pred.max(1)
        rate += predicted.eq(y).sum().item()
    print('loss: %.3f | acc: %.3f%%'
            % ((training_loss/(BATCH_SIZE+1)), 100.*rate/(BATCH_SIZE+1)))

def test(dataloader,net,loss_fn):
    global best_acc
    test_loss = 0
    rate = 0
    with torch.no_grad():
        for batch,(X,y) in enumerate(dataloader):
            X,y = X.to(device), y.to(device)

            #compute prediction error
            pred = net(X)
            loss = loss_fn(pred,y)
            test_loss += loss.item()
            _, predicted = pred.max(1)
            rate += predicted.eq(y).sum().item()
    print('loss: %.3f | acc: %.3f%%'
            % ((test_loss/(BATCH_SIZE+1)), 100.*rate/(BATCH_SIZE+1)))

    # save the model
    acc = 100.*rate/(batch+1)
    if acc > best_acc:
        print("saved pytorch model state to model.pth")
        state = {
            'net': net.state_dict(),
            'acc': acc,
        }
        if not os.path.isdir('newcheckpoint'):
            os.mkdir('newcheckpoint')
        torch.save(state, './newcheckpoint/ckpt.pth')
        best_acc = acc

# train by epoch
for i in range(EPOCHS):
    train(trainloader,net,loss_fn,optimizer)
    test(trainloader,net,loss_fn,optimizer)
    print(f'Epoch {i+1}\n----------')
print('Done!')

