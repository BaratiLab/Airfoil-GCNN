#%%

from hyperopt import hp, tpe, STATUS_OK, Trials
from hyperopt.fmin import fmin
from hyperopt.pyll import scope
import numpy as np 
import sklearn.gaussian_process as gp 
import sklearn.model_selection as ms
import os
from matplotlib import pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_validate
from sklearn.linear_model import Lasso 
from sklearn.linear_model import Ridge 
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import KFold
import pandas as pd
import os
import torch.optim as optim
from sklearn import svm
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import math
import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F
import torch.nn
import tensorflow as tf
from tensorflow import keras






from torch.optim.lr_scheduler import MultiStepLR

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F







params = { #'font.family'      : 'serif',
           'font.size'        : 12,
           'xtick.labelsize'  : 'small',
           'ytick.labelsize'  : 'small',
           #'axes.labelsize'   : 'large',
           'axes.linewidth'   : 1.3 }
plt.rcParams.update(params)



labels = pd.read_csv("./data/labels.csv")
print(labels.iloc[:, 1])
maximum = np.max(labels.iloc[:,1])
minimum = np.min(labels.iloc[:, 1])
drag  = labels.iloc[:, 1] 

# Load Data
# ...



params = { #'font.family'      : 'serif',
           'font.size'        : 12,
           'xtick.labelsize'  : 'small',
           'ytick.labelsize'  : 'small',
           #'axes.labelsize'   : 'large',
           'axes.linewidth'   : 1.3 }
plt.rcParams.update(params)
testnodes = []
trainnodes = []
r2nodes = []
nodes = 700
X = np.zeros((0, 2*nodes))
c = 0
lim = 1500
y = np.zeros(0)
for label in labels.iloc[0:lim,0]:
    if label == 'fx79w660a':
        c = c+1
        continue
    adjvel = np.loadtxt('./data/' + str(nodes)+ 'manadjvel/' + label  + '_adjvel', dtype = 'double').reshape(1,-1)


    meanorig = np.mean(labels.iloc[0:lim, 1])
    stdorig = np.std(labels.iloc[0:lim, 1])
    if (labels.iloc[c, 1] > meanorig + 2*stdorig):
        print("drag too high, filtered")
        c = c+1
        continue
    if (labels.iloc[c, 1] < meanorig - 3*stdorig):
        print(meanorig - 4*stdorig)
        print(labels.iloc[c, 1], c)
        print("drag too low, filtered")
        c = c+1
        continue
   
    else:
        X = np.vstack((X, adjvel))
        y = np.append(y,labels.iloc[c, 1])
    
    c = c+1

X_avg = np.zeros((X.shape[0] , X.shape[1]-5))
for i in range(2*nodes -5):
    X_avg[:,i] = np.mean(X[:,i:i+5], axis = 1)


mean = np.mean(y)
std = np.std(y)
ynorm = (y - mean)/std
y = ynorm
y_list = []
X_list = []
[X_list.append(X[i].reshape(1, 35,40)) for i in range(0, len(X))]

train_x_list = X_list[0:int(0.8*len(X_list))]
test_x_list = X_list[int(0.8*len(X_list)):]
[y_list.append(y[i].reshape(1, -1)) for i in range(0, len(y))]
train_y_list  = y_list[0:int(0.8*len(X_list))]#.reshape(1, -1)
test_y_list = y_list[int(0.8*len(X_list)):]#.reshape(1, -1)
#%%
class Net(nn.Module):
    def __init__(self, dropout = 0.5):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv64 = nn.Conv2d(64, 64, 3 )
        self.batchnorm64 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3 )
        self.conv128 = nn.Conv2d(128, 128, 3 )
        self.batchnorm128 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.batchnorm256 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3)
        self.conv6 = nn.Conv2d(256,512, 3)
        self.conv512 = nn.Conv2d(512,512, 3)
        self.batchnorm512 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(256*3, 4096)
        self.fc2 = nn.Linear(4096, 1000)
        self.fc3 = nn.Linear(1000, 1)
        self.dropout = dropout
    def forward(self, x):

        x = self.batchnorm64(F.relu(self.conv1(x)))

        x = self.batchnorm64(F.relu(self.conv64(x)))
 
        x = self.pool(x)

        x = self.batchnorm128(F.relu(self.conv2(x)))

        x = self.batchnorm128(F.relu(self.conv128(x)))
        
        x = self.pool(x)

        x = self.batchnorm256(F.relu(self.conv3(x)))

        x = self.batchnorm256(F.relu(self.conv4(x)))

        x = x.view(-1, 256*3)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, 1, 1)


dropout = 0.4
weight_decay = 0.2
momentum= 0.9
lr = 0.001
data_list_train = TensorDataset(torch.FloatTensor(train_x_list), torch.FloatTensor(train_y_list))
data_list_test = TensorDataset(torch.FloatTensor(test_x_list), torch.FloatTensor(test_y_list))
test_loader = DataLoader(data_list_test)
train_loader = DataLoader(data_list_train, batch_size =16)
net = Net(dropout = dropout )
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net.to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr = lr , momentum = momentum  , weight_decay=weight_decay) #
scheduler = MultiStepLR(optimizer, milestones=[50,100,150], gamma=0.1)
test_loader = DataLoader(data_list_test, batch_size = 16)
train_loader = DataLoader(data_list_train, batch_size =16)

for epoch in range(2000): #2000
    running_loss = 0.0
    count = 0
    mae = 0
    correct = 0
    preds = np.zeros(0)
    ys = np.zeros(0)
    loss_all = 0
    l1loss_all = 0
    preds = np.zeros(0)
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        pred = outputs

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        #print statistics
        running_loss += loss.item()
        if i % 10 == 9: # print every 2 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i+1, running_loss))
            running_loss = 0.0

        if (epoch % 1== 0):
            count = count + 1
            preds = np.append(preds, (pred.data).cpu().numpy())
            ys = np.append(ys, (data[1].cpu().numpy()))

        correct = torch.abs(pred.cpu() - data[1].cpu()).mean()
        mae = mae + correct
    
    mean_ys = np.mean(ys)
    sstot = np.sum((ys - mean_ys)**2)
    sse = np.sum((np.array(preds) - ys)**2)
    r2 = 1 - sse/sstot
    print('[%d, %5d] R2: %.3f' % (epoch + 1, i+1, r2))
    running_loss = 0.0
    count = 0
    mae = 0
    correct = 0
    preds = np.zeros(0)
    ys = np.zeros(0)
    loss_all = 0
    l1loss_all = 0
    preds = np.zeros(0)
    scheduler.step()

    for i, data in enumerate(test_loader, 0):
       
        inputs, labels = data[0].to(device), data[1].to(device)

        outputs = net(inputs)
        pred = outputs
        loss = criterion(outputs, labels) 

        running_loss += loss.item()

        if i % 10 == 9: # print every 2 mini-batches
            print('[%d, %5d] test loss: %.3f' % (epoch + 1, i+1, running_loss))
            running_loss = 0.0
        if (epoch % 1== 0):
            count = count + 1
            preds = np.append(preds, (pred.data).cpu().numpy())
            ys = np.append(ys, (data[1].cpu().numpy()))

        correct = torch.abs(pred.cpu() - data[1].cpu()).mean()
        mae = mae + correct
    
    mean_ys = np.mean(ys)
    sstot = np.sum((ys - mean_ys)**2)
    sse = np.sum((np.array(preds) - ys)**2)

    r2 = 1 - sse/sstot
    print('[%d, %5d] Test R2: %.3f' % (epoch + 1, i+1, r2))





print('Finished Training')

