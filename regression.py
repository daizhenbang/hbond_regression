import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random
'''Plot the fitting'''
file = open('allDataTrimmed.txt','r');
lines = file.readlines();
file.close();

#file2 = open('std_edge.txt','r');
#lines2 = file2.readlines();
#file2.close();

numSample = len(lines);
numColumn = len(lines[0].split());

allX = np.zeros((numSample,numColumn-1));
Y = np.zeros((numSample,1));
#
for i in range(numSample):
    temp = lines[i].split();
    
    for j in range(numColumn-1):
        allX[i][j] = temp[j];
    Y[i] = temp[-1];

scaler = MinMaxScaler();
scaler.fit(allX);
scaledX = scaler.transform(allX);
#
reg = LinearRegression();
reg.fit(scaledX,Y);

predY = reg.predict(scaledX);
print(reg.coef_)
print(reg.score(scaledX,Y));

fig1 = plt.figure(1,figsize=(10,8));
plt.plot(Y,Y);
plt.scatter(Y,predY,marker='.')
plt.xlabel('DFT Energy/Ry',fontsize=18)
plt.ylabel('Fitted Energy/Ry',fontsize=18)
plt.show();
#plt.savefig('Fitting_reuslt.png',dpi=500)
#

'''End of plotting the fitting'''

'''Statistics of the dipoles and quadrupoles'''
#file2 = open('lattice_dipole.txt','r');
#file3 = open('lattice_quadrupole.txt','r');
#temp_dipole = file2.readlines();
#temp_quad = file3.readlines();
#file2.close();
#file3.close();
#
#numLength = len(temp_dipole);
#dipoles = np.zeros((numLength,3));
#quad = np.zeros((numLength,9));
#normDipoles = np.zeros((numLength,1));
#
#for i in range(numLength):
#    temp1 = temp_dipole[i].split();
#    temp2 = temp_quad[i].split();
#    dipoles[i] = temp1;
#    quad[i] = temp2;
#    normDipoles[i] = np.linalg.norm(dipoles[i]);
#    
#fig2 = plt.figure(2,figsize=(20,16));
#
##plt.hist(normDipoles,30);
##plt.title('Magnitude of the Lattice Dipoles');
##plt.ylabel('Count');
##plt.savefig('Magnitude_lattice_dipole.png',dpi=1000);
#for m in range(9):
#    i = random.randrange(0,numLength/8);
#    fig2.add_subplot(3,3,m+1)
#    start = 0+i*8; end = (i+1)*8;
#    plt.scatter(range(8),normDipoles[start:end])
#    plt.title('Dipoles of ' +str(i+1) + 'th Structure',fontsize=24)

#fig2.savefig('dipoles_of_selected_structure.png',dpi000);

'''Hisogram of the total energy'''
#file = open('dft.txt','r');
#temp = file.readlines();
#file.close();

#dft = np.zeros((len(temp),1));
#for i in range(len(temp)):
#    dft[i] = float(temp[i]);

#fig = plt.figure(1);
#plt.hist(dft,10);
