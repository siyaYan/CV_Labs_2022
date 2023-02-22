# -*- coding: utf-8 -*-
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data

torch.manual_seed(17)

# train_imgs_file = 'data/kmnist-npz/kmnist-train-imgs.npz'
# train_labels_file = 'data/kmnist-npz/kmnist-train-labels.npz'
batchsize = 50
epochs = 50
init_lr = 0.001

#   output[channel] = (input[channel] - mean[channel]) / std[channel]
#   output range(-1,1)<<-- input(0.5(mean),0.5(std))
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    # when mean=0.5,std=0.5 (inputmin(0)-0,5)/0.5=-1 & (inputmax(1)-0.5)/0.5=1
    # so output data range in (-1,1)    
    transforms.Normalize([0.5],[0.5]),
    # flip left and right    
    transforms.RandomHorizontalFlip(),
    # randomly crop image to 28 X 28 with 4 padding    
    transforms.RandomCrop(28,padding=4),
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5]),
])

import torch.utils.data as data
import pandas as pd
class MyDataset(data.Dataset):
    def __init__(self,imgs,labels, transform=None, target_transform=None):
        self.imgs = np.load(imgs)['arr_0']
        self.labels=np.load(labels)['arr_0']
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.imgs[idx,:,:]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
training_set = MyDataset(
    imgs='data/kmnist-npz/kmnist-train-imgs.npz',
    labels='data/kmnist-npz/kmnist-train-labels.npz',
    transform=train_transforms,
    target_transform=None

)
testing_set = MyDataset(
    imgs='data/kmnist-npz/kmnist-test-imgs.npz',
    labels='data/kmnist-npz/kmnist-test-labels.npz',
    transform=test_transforms,
    target_transform=None,
)
valing_set = MyDataset(
    imgs='data/kmnist-npz/kmnist-val-imgs.npz',
    labels='data/kmnist-npz/kmnist-val-labels.npz',
    transform=val_transforms,
    target_transform=None
)

trainloader = data.DataLoader(training_set, batch_size=batchsize, shuffle=True)
testloader = data.DataLoader(testing_set, batch_size=batchsize, shuffle=True)
valloader = data.DataLoader(valing_set, batch_size=batchsize, shuffle=True)

import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # convolutional layer(change channels' num)
        # H(output)=(H(input)−F+2P)/S+1
        # first conv:input shape (1,28,28)-> output shape (32,28,28)
        self.conv1 = nn.Conv2d(1,32,5,stride=1,padding=2)
        
        # first pooling:(32,28,28)->(32,14,14)
        
        # second conv: after pooling,input shape is (32,14,14) -> output shape (64,14,14)
        self.conv2 = nn.Conv2d(32,64,3,stride=1,padding=1)
        
        self.dropout = nn.Dropout(0.2)
        # second pooling:(64,14,14)->(64,7,7)
        
        # first linear connect: after second pooling,input shape is (64*7*7) -> output 1024
        self.fc1 = nn.Linear(7*7*64, 1024)
        # second linear connect:input 1024 -> output 10      
        self.fc2 = nn.Linear(1024 , 10)    
    
    def forward(self,x):   
    
    #  do first relu activation & pooling
        # first conv: output shape (batchsize,32,28,28)   
        res1 = F.relu(self.conv1(x))
        # pooling layer(change image size, usually 1/2)
        # H(output)=(H(input)−F)/S+1
        # first pooling: input shape (batchsize,32,28,28) -> output shape (batchsize,32,14,14)
        res2 = F.max_pool2d(res1,2)     
    
    #  second relu & pooling
        # second conv: output shape (batchsize,64,14,14)       
        res3 = F.relu(self.dropout(self.conv2(res2)))
        # second pooling: input shape (batchsize,64,14,14) -> output shape (batchsize,64,7,7)
        res4 = F.max_pool2d(res3,2)
   
    #  full connectings & relu
        # (batchsize,64,7,7) -> output shape (batchsize,64*7*7) 
        res4 = res4.view(res4.shape[0], -1)
        # first full linear connection:(batchsize,64*7*7)->(batchsize,1024)       
        res5 = self.fc1(res4)
        # relu activation function        
        res6 = F.relu(res5)
        # second full linear connection:(batchsize,1024)->(batchsize,10)
        output = self.fc2(res6)      
        return output

def training(epochs,trainloader,valloader):
    train_losses=[]
    train_accs=[]
    val_accs=[]
    val_losses=[]
    CNN = Model()
    
    # use crossentroyLoss as loss function    
    loss_func = nn.CrossEntropyLoss()
    # set Adam optimizer with learning rate 0.001 and betas in (0.9,0.999)   
    optimizer = torch.optim.Adam(CNN.parameters(), lr=1e-3, betas=(0.9,0.999))
    
    running_loss=0.0
    train_acc=0.0
    val_acc=0.0
    lr=init_lr #0.001
    
    for epoch in range(epochs):
        running_loss=0.0
        accuracy=0.0
        train_acc=0.0
        total=0.0
        train_loss=0.0
        eachtotal=0.0
        
#         different learning rate with more epochs
        if epoch > 20:
            if epoch < 30:
                lr=0.5*init_lr #0.0005
            else:
                if epoch < 40:
                    lr=0.3*init_lr #0.0003
                
                elif epoch <45:
                    lr=0.1*init_lr #0.0001
                else:
                    lr=0.05*init_lr #0.00005
                    
        epochs = 50
        lr = 0.001
        for i ,data in enumerate(trainloader,0):
            inputs,labels = data
            
            optimizer.zero_grad()
            
            outputs = CNN(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss +=loss.item()
            train_loss += running_loss
            accuracy += (outputs.argmax(dim=1) == labels).sum().item()
            train_acc += accuracy
            eachtotal+=labels.size(0)
            total+=eachtotal
#             writer.add_scalar("acc vs epoch", trainAccSum/n, epoch)
            if i%100==100-1:
#                 writer.add_scalar("acc vs epoch", trainAccSum/n, epoch)
#                 print(f'running_loss:{running_loss}, acc:{trainAccSum/n}')
                print(running_loss)
                print(f'[{epoch +1},{i+1:5d}], acc:{100.*accuracy/eachtotal:.2f}% ,loss:{running_loss/eachtotal:.5f}')
                running_loss=0.0
                accuracy=0.0
                eachtotal=0.0
        
        PATH = './checkpoints/cifar_net_{:02d}.pth'.format(epoch)
        torch.save(CNN.state_dict(),PATH)
        
#         writer.add_scalar("acc vs epoch", train_acc/len(trainloader), epoch)
        train_accs.append(100*train_acc/total)
        print(f'training acc of epoch{epoch+1:2d}: {100.*train_acc/total:.2f}%')
        train_losses.append(train_loss/total)
        print(f'training loss of epoch{epoch+1:2d}: {train_loss/total:.5f}')
        
        validation_loss = 0.0
        running_loss = 0.0
        val_acc=0.0
        total=0.0
        print('starting validation for epoch {}'.format(epoch+1))
        with torch.no_grad():
            for data in valloader:
                inputs,labels = data            
                outputs = CNN(inputs)
                loss = loss_func(outputs, labels)
#                 validation_loss = loss.item() * len(data)
                validation_loss+=loss.item()
                val_acc += (outputs.argmax(dim=1) == labels).sum().item()
                total+=len(data[1])
        
            val_accs.append(100*val_acc/total)
            val_losses.append(validation_loss/total)
            print(f'validation loss for epoch{epoch+1:2d}: {validation_loss/total:.5f}')
            print(f'validation acc for epoch{epoch+1:2d}: {100.*val_acc/total:.2f}%')
            
    print('finished training')
    return train_accs,train_losses,val_accs,val_losses

# modify
def testing(testloader,epochs):
#     test_losses=[]
    test_accs=[]
    CNN = Model()
    for epoch in range(epochs):   
        CNN.load_state_dict(torch.load('./checkpoints/cifar_net_{:02d}.pth'.format(epoch)))
    
        correct=0.0
        total =0 
    
        with torch.no_grad():
            for data in testloader:
                images,labels = data
                outputs = CNN(images)
                correct += (outputs.argmax(dim=1) == labels).sum().item()
#             _,predicted=torch.max(outputs,data,1)
                total+=labels.size(0) 
#             correct+=(predicted==labels).sum().item()
        test_accs.append(100*correct/total)
        print(f'accuracy of the cnn epoch {epoch+1} on the testing images is :{100.*correct/total:.2f}%')
    
    return test_accs
            from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
writer2 = SummaryWriter()

def display1(train_accs,train_losses,val_accs,val_losses):
    for i,train_acc in enumerate(train_accs,0):
        
        writer.add_scalar("train_acc vs epoch", train_acc, i+1)
        writer.add_scalar("train_loss vs epoch", train_losses[i], i+1)    
        writer.add_scalar("validation_acc vs epoch", val_accs[i], i+1)
        writer.add_scalar("validation_loss vs epoch", val_losses[i], i+1)
        
def display2(train1_accs):
    for i,train_acc in enumerate(train1_accs,0):
        writer.add_scalar("accuracies of different lr(0.001vs0.0001)", train_acc, i+1)
        
def display3(train2_accs):
    for i,train_acc in enumerate(train2_accs,0):
        writer.add_scalar("accuracies of different lr(0.001vs0.0001)", train_acc, i+1)
        
# def displayModify(val_accs,val_losses):
#     for i,val_acc in enumerate(val_accs,0):  
#         writer.add_scalar("validation_acc vs epoch", val_acc, i+1)
#         writer.add_scalar("validation_loss vs epoch", val_losses[i], i+1)

def display4(train1_accs):
    for i,train1_acc in enumerate(train1_accs,0):
        writer2.add_scalar("lr=0.001~0.00005_vs_lr=0.0005", train1_acc, i+1)

def display5(train2_accs):
    for i,train2_acc in enumerate(train2_accs,0):
        writer2.add_scalar("lr=0.001~0.00005_vs_lr=0.0005", train2_acc, i+1)        
        
train_accs,train_losses,val_accs,val_losses=training(epochs,trainloader,valloader)

test_accs=testing(testloader,epochs)

display(train_accs,train_losses,val_accs,val_losses)

