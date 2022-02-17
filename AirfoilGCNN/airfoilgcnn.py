#%% Import packages


import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.data import Data,DataLoader
from torch_geometric.nn import GraphConv, TopKPooling,  GCNConv, avg_pool, TAGConv, SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import tqdm
from tqdm import tqdm
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Prepare for data loading
num_epochs = 500

data_list_train = []
data_list_test = []
data_list = []
datachoice = input("dataset? Choices are 'zeroAoA', 'closestzeroAoA', 'shuffledzeroAoA' and 'multipleAoA'")
plotbool = input("Plot output? (Y/N)")
if plotbool == "Y" or plotbool == "y":
    plotbool = True
elif plotbool == "N" or plotbool == "n":
    plotbool = False
else:
    print("Did not recognize, will not plot")

if datachoice == "zeroAoA" or datachoice == "shuffledzeroAoA" or datachoice == 'closestzeroAoA':
    labels = pd.read_csv("./data/labels.csv")
elif datachoice == "multipleAoA":
    labels = pd.read_csv("./data/labels_large.csv")
else: 
    raise ValueError('dataset choice not recognized, try again.')
maximum = np.max(labels.iloc[:,1])
minimum = np.min(labels.iloc[:, 1])
c = 0
print(labels)

#%%
d = 0
lim = int(input("How many airfoils do you want to train on? (Max is 1500 for zeroAoA and shuffledzeroAoA and 10000 for multipleAoA)"))
draglist = []
mean = np.mean(labels.iloc[0:lim, 1])
std = np.abs(np.std(labels.iloc[0:lim, 1]))
topk = 0.5


#Filter out outlier airfoils, outside -3 STD to 3 STD from the mean
for label in labels.iloc[:, 0]:
    if label == 'fx79w660a':
        d= d + 1
        continue
    if (labels.iloc[d, 1] > mean + 3*std):
        print("drag too high, filtered")
        d = d+1
        continue
    if (labels.iloc[d, 1] < mean - 3*std):
        print(labels.iloc[d, 0], d)
        print("drag too low, filtered")
        d = d+1
        continue
    draglist.append(labels.iloc[d, 1])
    d = d+1


dragarray = np.array(draglist)
mean = np.mean(dragarray)
std = np.std(dragarray)

#Normalize drag
drag  = (labels.iloc[:, 1] - mean)/(std)
print("mean of drag", np.mean(drag))
print("std of drag", np.std(drag))

omit = 0
###################
names =[]
###################
broken_labels = []
#Load in Data
for label in tqdm(labels.iloc[:,0]):
    afn = label.split('angle')[0]
    
    #These two airfoils don't work
    if label == 'fx79w660a' or label == 'e558angle4':
        omit += 1
        c = c+1
        continue
    meanorig = np.mean(labels.iloc[0:lim, 1])
    stdorig = np.abs(np.std(labels.iloc[0:lim, 1]))
    if (labels.iloc[c, 1] > meanorig + 3*stdorig):
        print(labels.iloc[c, 0], c)
        print("drag too high, filtered")
        omit += 1
        c = c+1
        continue
    if (labels.iloc[c, 1] < meanorig - 3*stdorig):
        print(labels.iloc[c, 0], c)
        print("drag too low, filtered")
        omit +=1
        c = c+1
        continue
    
    try:
        if datachoice == "closestzeroAoA":
            vel = np.loadtxt('./data/1000manpool_vel/' + label + '_vel_pool', dtype = 'double')
            edge = np.loadtxt('./data/1000manpool_edge_fixed/' + label + '_edge_pool_fixed', dtype = 'double')
        if datachoice == "shuffledzeroAoA":
            vel = np.loadtxt('./data/900randpool_vel/' + label + '_vel_pool', dtype = 'double')
            edge = np.loadtxt('./data/900randvertpool_edge_fixed/' + label + '_edge_pool_fixed', dtype = 'double')
        if datachoice == "zeroAoA":
            vel = np.loadtxt('./data/velocity/' + label + '_vel', dtype = 'double')
            edge = np.loadtxt('./data/edges/' + label + '_edges', dtype = 'double')
        if datachoice == "multipleAoA":
            vel = np.loadtxt('./data/velocity_large/' + label + '_vel', dtype = 'double')
            edge = np.loadtxt('./data/edges_large/' + label + '_edges', dtype = 'double')
        if not (int(np.max(edge)) == len(vel) - 1):
            broken_labels.append(label)
            c = c+1
            continue
        veltensor = torch.from_numpy(vel).double()
        edgetensor = torch.from_numpy(edge.T).long()
        ytensor = torch.from_numpy(np.array((labels.iloc[c, 1] - mean)/(std)))  # drag[c]
        data = Data(x = veltensor, edge_index = edgetensor,y= ytensor.unsqueeze(0))
    except Exception as e:        
        print(e, label, "this one broke, (several are fine, if this prints out for every airfoil, check your filesystems)")
        raise Exception
        c = c+1
        continue
    # Train Test Split
    a = np.random.random_sample()
    if a < 0.8:
        data_list_train.append(data)
    else:
        data_list_test.append(data)
        ##############################
        names.append(label)
        ##############################
    data_list.append(data)

    if (c % 50 == 0):
        print(c, "Airfoils loaded out of", lim)
    if (c > lim):
        break

    c = c+1

test_dataset = data_list_test
train_dataset = data_list_train
dataset = data_list
test_loader = DataLoader(data_list_test) 
train_loader = DataLoader(data_list_train, batch_size =16)
print(len(data_list), "data size")
print(omit*100/lim, "% of data omitted")

###############
df = pd.DataFrame(names)
df.to_csv('names.csv')
##############
#%%
class Net(torch.nn.Module):
    def __init__(self):

        super(Net, self).__init__()
        conv_width = 64
        self.conv1 =  SAGEConv(2, conv_width)
        self.pool1 = TopKPooling(conv_width, ratio=topk)
        self.conv2 =  SAGEConv(conv_width, conv_width)
        self.pool2 = TopKPooling(conv_width, ratio= topk)
        self.conv3 =  SAGEConv(conv_width, conv_width)
        self.pool3 = TopKPooling(conv_width, ratio=topk)
        self.conv4 =  GCNConv(conv_width, conv_width)
        self.pool4 = TopKPooling(conv_width, ratio=topk)
        self.lin1 = torch.nn.Linear(2*conv_width, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 1)

    def forward(self, data):
        """
        data: Batch of Pytorch Geometric data objects, containing node features, edge indices and batch size
            
        returns: Predicted normalized drag value
        """
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        #print(x.shape)
        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv4(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool4(x, edge_index, None, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = x1+x2#+x3#+x4

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.0, training=self.training)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x
#%%

device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu') #torch.device('cpu')   
model = Net().to(device).float()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005) 


def train(epoch):
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, data.y.view(len(output), 1).float())
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)

test_mse = []
train_mse = []

def test(loader):
    model.eval()
    mae = 0
    correct = 0
    preds = np.zeros(0)
    ys = np.zeros(0)
    loss_all = 0
    l1loss_all = 0
    for data in loader:
        data = data.to(device)
        pred = model(data)
        count = 0 
        loss = F.mse_loss(pred, data.y.view(len(pred), 1).float())
        l1loss = torch.nn.L1Loss() 
        maeloss = l1loss(pred, data.y.view(len(pred), 1).float())
        loss_all += data.num_graphs * loss.item()
        l1loss_all += data.num_graphs * maeloss.item()
        if (epoch % 1== 0):
            count = count + 1
            preds = np.append(preds, (pred.data).cpu().numpy())
            ys = np.append(ys, (data.y.data).cpu().numpy())
        correct = torch.abs(pred - data.y).mean()
        mae = mae + correct

    mean_ys = np.mean(ys)
    sstot = np.sum((ys - mean_ys)**2)
    sse = np.sum((np.array(preds) - ys)**2)
    r2 = 1 - sse/sstot

    #print("R2", r2)
    return loss_all / len(loader.dataset), r2, l1loss_all/len(loader.dataset), preds, ys

if not(os.path.exists('./results')):
    os.makedirs('results', exist_ok=True)
for epoch in tqdm(range(1, num_epochs)):

    loss = train(epoch)    
    test_acc, testr2, testmae, testpreds, testys = test(test_loader)
    train_acc, trainr2, trainmae, trainpreds, trainys = test(train_loader)
    test_mse.append(test_acc)
    train_mse.append(train_acc)
    if plotbool:                
        data = testys
        pred = testpreds
        datasort = np.sort(data)
        dataidx = np.argsort(data)



        plt.plot(datasort, pred[dataidx], 'r.', ms = 2)
        plt.plot(np.arange(-4, 4), np.arange(-4,4), 'k-')
        plt.xlabel("Actual Drag Force")
        plt.ylabel("Predicted Drag Force")
        plt.title("GCNN - Test Set")
        plt.ylim(-4, 2)
        plt.xlim(-4, 2)
        plt.pause(1e-10)
        plt.clf()

    print('Epoch: {:03d}, Loss: {:.5f}, Train MSE: {:.5f}, Test MSE: {:.5f}, Train MAE: {:.5f}, Test MAE: {:.5f}, Train R^2: {:.5f}, Test R^2: {:.5f}'.
          format(epoch, loss, train_acc, test_acc, trainmae, testmae, trainr2, testr2))

    if (np.mod(epoch+1,10) == 0):
        ##########################
        np.savetxt('results/gcnn_actual_train_m_epoch{}.csv'.format(epoch) , trainys , delimiter=',') 
        np.savetxt('results/gcnn_pred_train_m_epoch{}.csv'.format(epoch) , trainpreds , delimiter=',') 

        np.savetxt('results/gcnn_actual_test_m_epoch{}.csv'.format(epoch) , testys , delimiter=',') 
        np.savetxt('results/gcnn_pred_test_m_epoch{}.csv'.format(epoch) , testpreds , delimiter=',') 
        ###########################

plt.plot(np.array(test_mse))
plt.plot(np.array(train_mse))

plt.legend(["Test", "Train"])
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.show()

