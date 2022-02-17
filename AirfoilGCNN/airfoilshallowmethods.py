
import numpy as np 
import sklearn.gaussian_process as gp 
from matplotlib import pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso 
from sklearn.linear_model import Ridge 
import pandas as pd
import tqdm
from tqdm import tqdm as tqdm
from sklearn import model_selection as ms

params = {
           'font.size'        : 12,
           'xtick.labelsize'  : 'small',
           'ytick.labelsize'  : 'small',
           'axes.linewidth'   : 1.3 }
plt.rcParams.update(params)



labels = pd.read_csv("./labels.csv")
print(labels.iloc[:, 1])
maximum = np.max(labels.iloc[:,1])
minimum = np.min(labels.iloc[:, 1])
drag  = labels.iloc[:, 1]

shuffledtoggle = input("Do you want to shuffle the node ID's (shuffled: 'y', unshuffled: 'n') \n")
if not shuffledtoggle == 'y' and not shuffledtoggle == 'n':
    raise Exception("Didn't recognize, try again")
modeltoggle = input("What model do you want to use? \n Choices: 'GPR' (Gaussian Process Regression), 'SVR' (Support Vector Regression), 'RF' (Random Forest), 'GB' (Gradient Boosting), 'Ridge' (L2 regression), 'Lasso' (L1 regression) \n")

params = {
           'font.size'        : 12,
           'xtick.labelsize'  : 'small',
           'ytick.labelsize'  : 'small',
           'axes.linewidth'   : 1.3 }
plt.rcParams.update(params)
testnodes = []
trainnodes = []
r2nodes = []
nodes = 1200

X = np.zeros((0, 2*nodes))
c = 0
lim = int(input("How many airfoils do you want to train on? \n"))
y = np.zeros(0)
# Load Data
for label in tqdm(labels.iloc[0:lim,0]):
    if label == 'fx79w660a':
        c = c+1
        continue
    adjvel = np.loadtxt('./' + str(nodes)+ 'manadjvel/' + label  + '_adjvel', dtype = 'double').reshape(1,-1)

    meanorig = np.mean(labels.iloc[0:lim, 1])
    stdorig = np.std(labels.iloc[0:lim, 1])
    if (shuffledtoggle == 'y'):
        adjvel = np.random.permutation(adjvel[0])
    
    if (labels.iloc[c, 1] > meanorig + 3*stdorig):
        print(labels.iloc[c, 0], "drag too high, filtered")
        c = c+1
        continue
    if (labels.iloc[c, 1] < meanorig - 3*stdorig):
        print(labels.iloc[c, 0], "drag too low, filtered")
        c = c+1
        continue
   
    else:
        X = np.vstack((X, adjvel))
        y = np.append(y,labels.iloc[c, 1])    
    c = c+1


mean = np.mean(y)
std = np.std(y)
ynorm = (y - mean)/std
y = ynorm

def model_run(n_estimators, max_depth):
    X_train,_, y_train,_ = ms.train_test_split(X, y, test_size = 0.2, random_state = 0)
    if modeltoggle == 'RF':
        model = RandomForestRegressor(n_estimators = 500, max_depth = 3, random_state = 0)
    elif modeltoggle == 'GPR':
        kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3))*gp.kernels.RBF(10.0, (1e-3, 1e3))
        model = gp.GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 10, alpha = 0.1, normalize_y = True)
    elif modeltoggle == 'SVR':
        model = SVR(gamma = 'scale', C = 1.0, epsilon = 0.2)
    elif modeltoggle == 'Ridge':
        model = Ridge(alpha = 2000.0)
    elif modeltoggle == 'Lasso':
        model = Lasso(alpha = 0.01)
    elif modeltoggle == 'GB':
        model = GradientBoostingRegressor(n_estimators = n_estimators, max_depth = max_depth, random_state = 0) # 500 , 1 respectively
    model.fit(X_train, y_train)
    scores = ms.cross_validate(model, X,y, cv = 5, scoring = ('r2', 'neg_mean_squared_error'), return_train_score = True)
    mse_mean = scores["test_neg_mean_squared_error"].mean()
    mse_std = scores["test_neg_mean_squared_error"].std()
    r2_mean = scores["test_r2"].mean()
    r2_std = scores["test_r2"].std()

    r2_train_mean = scores["train_r2"].mean()
    r2_train_std =  scores["train_r2"].std()
    mse_train_mean = scores["test_neg_mean_squared_error"].mean()
    mse_train_std = scores["test_neg_mean_squared_error"].std()

    print('Train MSE: {:.5f} ± {:.5f} , Test MSE: {:.5f} ± {:.5f}, Train R^2: {:.5f} ± {:.5f}, Test R^2: {:.5f} ± {:.5f}'.
            format(mse_train_mean, mse_train_std, mse_mean, mse_std, r2_train_mean, r2_train_std, r2_mean, r2_std))
    return r2_mean 

r2_test = model_run(n_estimators = 500, max_depth=1 )  
  
