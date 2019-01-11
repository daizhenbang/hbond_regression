import numpy as np
import matplotlib.pyplot as plt    
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import seaborn as sns


def readData(name):
    file = open(name,'r');
    lines = file.readlines();
    file.close();
    lenRow = len(lines);
    lenCoulom = len(lines[0].split());
    data = np.zeros((lenRow,lenCoulom));
    for i in range(lenRow):
        temp = lines[i].split();
        for j in range(lenCoulom):
            data[i][j] = float(temp[j]);
        
    return data;


font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)
      
dft = readData('dft.txt');      
deviations = readData('allDeviations.txt');
ewald = readData('ewald.txt');
nHBond = readData('allHBondStrength.txt')
bv = readData('ave_std_edge.txt');
allGlazer = readData('glazer.txt');
tilting = readData('tiltingAngle.txt');
tiltingSquare = readData('tiltingAngleSquare.txt');
poles = readData('poles_coupling_charge_center.txt');
p4 = np.reshape(allGlazer[:,1],[200,1]);
dd = np.reshape(allGlazer[:,1],[200,1]);

'''Plot these quantities'''
#fig = plt.figure(figsize=(10,8));
#plt.hist(nHBond,20);
#plt.title('Average Number of H Bonds per Supercell',fontsize=20)
#plt.xlabel('Average Number of H Bonds',fontsize=20)
#plt.ylabel('Count',fontsize=20)
#command = input("Do you want to save the figure? y/n: ");
#if command == 'y':
#    fig.savefig('Hist_aveHBonds.png',dpi=500)


'''Correlation'''

X = np.concatenate([deviations,\
                    nHBond,\
                    ],axis=1);
allData = np.concatenate([X,dft],axis = 1);
#
'''Plot with Seaborn'''
sns.set(style="white")

# Generate a large random dataset
d = pd.DataFrame(data=allData);

# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


''''''

dft = (dft - np.min(dft))*13.6*1000;

sizeTrain = 160;
sizeFeature = len(X[0,:]);
sizeData = len(X[:,0]);

allData = np.concatenate((X,dft),axis=1);
np.random.shuffle(allData);

scaler = MinMaxScaler();
X = allData[:,0:sizeFeature];
scaler.fit(X);
X = scaler.transform(X);
dft = allData[:,-1];


'''Neural Network'''
#X_train = np.reshape(X[0:sizeTrain,:],[sizeTrain,sizeFeature]);
#X_test = np.reshape(X[sizeTrain:sizeData,:],[sizeData-sizeTrain,sizeFeature]);
#dftE_train = dft[0:sizeTrain];
#dftE_test = dft[sizeTrain:sizeData];
#
#neuralRegressor = MLPRegressor(hidden_layer_sizes=(40,40,40,40),activation='relu',solver='lbfgs',\
#                               max_iter=20000000,tol=1e-14);
#                               
#neuralRegressor.fit(X_train,dftE_train);
#predE_train = neuralRegressor.predict(X_train);
#predE_test = neuralRegressor.predict(X_test);
#print(neuralRegressor.score(X_train,dftE_train));
#print(neuralRegressor.score(X_test,dftE_test));
#
#fig = plt.figure(1,figsize=(10,4));
#fig.add_subplot(1,2,1);
#plt.plot(dftE_train,dftE_train);
#plt.scatter(dftE_train,predE_train,marker='.');
#
#fig.add_subplot(1,2,2);
#plt.plot(dftE_test,dftE_test);
#plt.scatter(dftE_test,predE_test,marker='.');
''''''

'''Linear Regression'''
regressor = LinearRegression();
dftE = dft;

regressor.fit(X,dftE);
predE = regressor.predict(X);

fig = plt.figure(1);
plt.plot(dftE,dftE);
plt.scatter(dftE,predE,marker='.');

print(regressor.coef_);
print(regressor.score(X,dftE));
''''''

