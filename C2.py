#!/usr/bin/env python
# coding: utf-8

# In[8]:


import wandb
import torch
import torchvision.transforms as T
import torchvision
import torch.nn as nn


# In[2]:


# !pip install --upgrade torch torchvision


# In[3]:


wandb.login()


# In[4]:


learning_rate = 0.001
epochs = 90
num_hidden_layers = 3
hidden_layer_sizes = [1000,800,800]
optimizer = "Adam"
loss = "cross-entropy"
architecture = "MLP"
run = "1"
dataset = "CIFAR-10"
batch_size = 32

wandb.init(
    project="SMAI-Project", 
    name=f"experiment_{architecture}_{dataset} {run}",
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
      "batch_size":batch_size
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
                T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)), # mean values and stds of RGB channels om cifar10
            ])
test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)) # mean values and stds of RGB channels om cifar10
])

# Load the dataset
trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True,
                                        download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


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
  for i in range(1,num_hidden_layers):
    model.add_module(f'hidden{i}', nn.Linear(hidden_layer_sizes[i-1],hidden_layer_sizes[i]))
    model.add_module(f'relu{i}', nn.ReLU())
  model.add_module('output', nn.Linear(hidden_layer_sizes[-1],p))
  return model


# In[17]:


def val_check_accuracy(data,model,loss_fn):
  num_samples = 0
  num_correct = 0
  num_batches = 0
  totalloss = 0
  with torch.no_grad():
    for x,y in data:
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


def train(model , optimiser , loss_fn):
    for epoch in range(epochs):
        model = model.to(device) # moving model on cuda
        for t,(x,y) in enumerate(trainloader):
            x = x.to(device)
            y = y.to(device , dtype=torch.long)
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
        trainacc , trainloss = val_check_accuracy(trainloader,model,loss_fn)
        wandb.log({"train loss": trainloss, "train accuracy": trainacc})
        print(f"Epoch {epoch+1} : train loss : {trainloss} train accuracy : {trainacc} val loss : {loss} val accuracy : {acc}")


# In[18]:


n = 3*32*32
p = 10

model = mlp(n , p , hidden_layer_sizes , num_hidden_layers)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train(model , optimizer , loss_fn)


# In[ ]:


wandb.finish()

