#!/usr/bin/env python
# coding: utf-8

# In[8]:


import wandb
import torch
import torchvision.transforms as T
import torchvision
import torch.nn as nn
import random
import numpy as np

import argparse
parser = argparse.ArgumentParser()

# In[2]:


# !pip install --upgrade torch torchvision


# In[3]:


wandb.login()


# In[4]:

parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--noise", type=float, default=0.0)
parser.add_argument("--early_stopping", type=bool, default=False)
parser.add_argument("--early_stopping_delta", type=float, default=0.001)
parser.add_argument("--early_stopping_patience", type=int, default=10)
parser.add_argument("--architecture" , default="MLP")
parser.add_argument("--num_convblocks" , type=int , default=3)
parser.add_argument("--input_size" , type=int , default=32)
parser.add_argument("--num_classes" , type=int , default=10)

args = parser.parse_args()
print(args)

learning_rate = 0.001
epochs = 1000
num_hidden_layers = 3
# hidden_layer_sizes = [int(1000//(1-args.dropout)),int(800//(1-args.dropout)),int(800//(1-args.dropout))]
hidden_layer_sizes = [1000,800,800]
optimizer = "Adam"
loss = "cross-entropy"
architecture = args.architecture
run = "1"
dataset = "CIFAR-10"
batch_size = 32
noise = args.noise



wandb.init(
    project="SMAI-Project", 
    name=f"experiment_{architecture}_{dataset}",
    # Track hyperparameters and run metadata
    config={
    "learning_rate": learning_rate,
    "architecture": architecture,
    "dataset": dataset,
    "epochs": epochs,
    "num_hidden_layers":num_hidden_layers,
    "hidden_layer_sizes" : hidden_layer_sizes,
    "optimizer":optimizer,
    "loss":loss,
    "batch_size":batch_size,
    "weight_decay":args.weight_decay,
    "dropout":args.dropout,
    "noise":args.noise,
    "early_stopping":args.early_stopping,
    "early_stopping_delta":args.early_stopping_delta,
    "early_stopping_patience":args.early_stopping_patience,
    "num_convblocks":args.num_convblocks,
    "input_size":args.input_size,
    "num_classes":args.num_classes
    })


# In[5]:


device = torch.device('cpu')
if torch.cuda.is_available():
  device = torch.device('cuda')
print(device)
dtype = torch.float32


# In[6]:


# Cite : https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# Some Data Augmentation
train_transform = T.Compose([
                T.RandomCrop(32,padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)), # mean values and stds of RGB channels on cifar10
            ])
test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)) # mean values and stds of RGB channels om cifar10
])

# Load the dataset
trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True,
                                        download=True, transform=train_transform)

# Add noise to the dataset
if args.noise > 0:
  lendataset = len(trainset.targets)
  indices = torch.randint(0,lendataset,(int(args.noise*(lendataset)),))
  for index in indices:
      trainset.targets[index] = random.randint(0,9)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=1)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[13]:


def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


# In[11]:


def mlp( n , p , hidden_layer_sizes , num_hidden_layers):
  '''n is the input features and p is the number of classes'''
  model = nn.Sequential()
  model.add_module('input', nn.Linear(n,hidden_layer_sizes[0]))
  model.add_module('relu0', nn.ReLU())
  model.add_module('dropout0', nn.Dropout(p=args.dropout))
  for i in range(1,num_hidden_layers):
    model.add_module(f'hidden{i}', nn.Linear(hidden_layer_sizes[i-1],hidden_layer_sizes[i]))
    model.add_module(f'relu{i}', nn.ReLU())
    model.add_module(f'dropout{i}', nn.Dropout(p=args.dropout))
  model.add_module('output', nn.Linear(hidden_layer_sizes[-1],p))
  return model

def AlexNet(input_size = 32 , num_classes =  10 , dropout = 0.5 , num_convblocks = 3):
    model = nn.Sequential()
    # model.add_module('conv1',nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1)) # (input_size , input_size , 3) -> (input_size , input_size , 64)
    # model.add_module('relu1',nn.ReLU())
    model.add_module('convblock0' , nn.Sequential(
        nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1) , 
        nn.ReLU() , 
        nn.MaxPool2d(kernel_size=2,stride=2) , # (input_size , input_size , 64) -> (input_size/2 , input_size/2 , 64)
    ))

    assert(num_convblocks > 0)
    assert(input_size % (2**num_convblocks) == 0) 

    for i in range(1,num_convblocks):
        model.add_module(f'convblock{i}' , nn.Sequential(
            nn.Conv2d(64*(2**(i-1)),64*(2**i),kernel_size=3,stride=1,padding=1) , 
            nn.ReLU() , 
            nn.MaxPool2d(kernel_size=2,stride=2) , # (input_size , input_size , 64*(2**(i-1))) -> (input_size/2 , input_size/2 , 64 * (2**i)))
        ))
    
    model.add_module('flatten' , nn.Flatten()) # (input_size/(2**num_convblocks) , input_size/(2**num_convblocks) , 64 * (2**(numblocks-1))) -> (input_size/(2**num_convblocks) * input_size/(2**num_convblocks) * 64 * (2**(numblocks-1)))
    model.add_module('dropout' , nn.Dropout(dropout))
    model.add_module('fc1' , nn.Linear(((input_size//(2**num_convblocks))**2 )*64 * (2**(num_convblocks-1))  , 128))
    model.add_module('relu1' , nn.ReLU())
    model.add_module('fc2' , nn.Linear(128 , num_classes))
    return model



# In[17]:


def val_check_accuracy(data,model,loss_fn):
  num_samples = 0
  num_correct = 0
  num_batches = 0
  totalloss = 0
  with torch.no_grad():
    for x,y in data:
      if args.architecture == "MLP":
            x = flatten(x)
      x = x.to(dtype=dtype,device=device)
      y = y.to(dtype=torch.long,device=device)
      scores = model(x)
      totalloss += loss_fn(scores,y).item()
      _,preds = scores.max(1)
      num_correct += (preds == y).sum()
      num_samples += preds.size(0)
      num_batches +=1
    acc = float(num_correct) / num_samples
    loss = float(totalloss)/num_batches
    return acc,loss


# In[16]:

class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def early_stop_fn(self, curr_val_loss):
        score = -curr_val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

early_stopping = EarlyStopping(patience=args.early_stopping_patience, delta=args.early_stopping_delta)     


def train(model , optimiser , loss_fn):
    for epoch in range(epochs):
        model = model.to(device) # moving model on cuda
        for t,(x,y) in enumerate(trainloader):
            x = x.to(device)
            y = y.to(device , dtype=torch.long)
            if args.architecture == "MLP":
                x = flatten(x)

            # Forward pass: compute predicted y by passing x to the model.
            preds = model(x)
            loss = loss_fn(preds, y)
            
            # Zero gradients, perform a backward pass, and update the weights.
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        # logging the loss and accuracy for each epoch
        acc,loss = val_check_accuracy(testloader ,model,loss_fn)
        wandb.log({"val loss": loss, "val accuracy": acc})
        # Early stopping
        if args.early_stopping:
            early_stopping.early_stop_fn(loss)
            if early_stopping.early_stop:
                print("Early stopping")
                print(f"Epoch Number: {epoch+1}")
                print(f"Early Stopping Validation Loss: {loss}")
                break
        trainacc , trainloss = val_check_accuracy(trainloader,model,loss_fn)
        wandb.log({"train loss": trainloss, "train accuracy": trainacc})
        print(f"Epoch {epoch+1} : train loss : {trainloss} train accuracy : {trainacc} val loss : {loss} val accuracy : {acc}")


# In[18]:


n = 3*32*32
p = 10

if args.architecture == "MLP":
    model = mlp(n , p , hidden_layer_sizes , num_hidden_layers)
elif args.architecture == "AlexNet":
   model = AlexNet(input_size = args.input_size , num_classes = args.num_classes , dropout = args.dropout , num_convblocks = args.num_convblocks)
else:
    raise Exception("Invalid Architecture")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)

train(model , optimizer , loss_fn)


# In[ ]:


wandb.finish()

