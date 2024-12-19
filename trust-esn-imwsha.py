
##################################################################
# 1 ) Get source data: IM-WSHA dataset
# Download from https://portals.au.edu.pk/imc/Pages/Datasets.aspx
# 
# Note: This requires some extra packages (requests
# and zipfile).
# You can also manually download and uncompress the data.
###################################################################

# Start the download    
print('downloading dataset ...')
import requests
url = 'https://portals.au.edu.pk/imc/Content/dataset/IM-WSHA_Dataset.zip'
r = requests.get(url)
with open('IM-WSHA_Dataset.zip', 'wb') as outfile:
    outfile.write(r.content)
print('download completed')

import zipfile
with zipfile.ZipFile('./IM-WSHA_Dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('./')

##################################################################
# 2 ) Reproduce the paper results
###################################################################

import reservoirpy as rpy

rpy.verbosity(0)  # no need to be too verbose here
rpy.set_seed(42)  # make everyhting reproducible !

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.io import loadmat
from sklearn.model_selection import train_test_split

# For article graphics set font to 24 points
plt.rcParams.update({'font.size': 18})

# Read data for Subject 1 (change this to use any other subject data)
df = pd.read_csv('./IM-WSHA_Dataset/IMSHA_Dataset/Subject 1/3-imu-one subject.csv')

# Correct the wrong labeling
df.loc[1150:1375,'activity_label']=1
df.loc[2390:2510,'activity_label']=2
df.loc[3300:3840,'activity_label']=3
df.loc[6000:6300,'activity_label']=5
df.loc[7200:7340,'activity_label']=6
df.loc[8400:8570,'activity_label']=7
df.loc[9675:9825,'activity_label']=8
df.loc[10900:11010,'activity_label']=9

# Get all the available features (3 accelerometers, 3 signals each)
features = df.keys()[1:] 

#####################################################################
# STEP 1: TRAIN THE ESN WITH A SUBSET OF ACTIVITIES
#####################################################################

# Train only with some of them, test the system with the rest
nt = 7 # number of activities to train with
train = np.arange(1,nt+1)
test = np.arange(nt+1,12)

# Prepare dataframes with test and train...
# Uncomment the following line to use all the available data for training
#df_train = df.loc[df['activity_label'].isin(train)]

# Uncomment the following block to remove initial and final data of 
# each activity.
df_train = pd.DataFrame()
for aa in train:
    df_tmp = df.loc[df['activity_label'] ==aa]
    df_train= pd.concat([df_train, df_tmp[300:-200]])

# and also input and output vectors
X_train = df_train[features].values.reshape(-1, len(features))
Y_train = df_train['activity_label'].values.reshape(-1, 1)

# Sample period
tm = 1/20.

# Create the ESN and set hyperparameters -- THESE HAVE NOT BEEN OPTIMIZED IN ANY WAY
from reservoirpy.nodes import Reservoir, Ridge, Input
n_states = 300
rho=0.95 
sparsity=0.01
Lr=0.025*2
Win_scale=150
input_scale = 1
Warmup = 20
set_bias = True 

print('Creating ESN...')
data = Input()
reservoir = Reservoir(n_states, lr=Lr, sr=rho, input_scaling=input_scale, rc_connectivity=sparsity, Win=rpy.mat_gen.bernoulli(input_scaling = Win_scale))
readout = Ridge(ridge=1e-7, input_bias = set_bias)
esn_model =  data >> reservoir >> readout
print(esn_model.node_names)

# Train the ESN with the train data
print('Training ESN...')
esn_model = esn_model.fit(X_train, Y_train, warmup=Warmup)
print(reservoir.is_initialized, readout.is_initialized, readout.fitted)


#####################################################################
# STEP 2: TEST THE METHOD - CREATE THE APPROXIMATED PDF
#####################################################################

C = []
T = []

# Run signal and get the reservoir state
print('Running train signal over reservoir...')
states = reservoir.run(X_train)

# Apply our method using the SVD decomposition and a sliding window
Q = Y_train.shape[0] # Total size
t = np.arange(Q).reshape(-1,1)*tm # time vector
S = 20  # Stride
L = 140 # Window lenght

print(f'Decomposing (wnd siz {L}, stride {S})...')
rango = np.arange(0,Q-L,S)
for i in rango:
    idx = np.arange(i,i+L)
    print(f"\rWindow {i} of {rango[-1]}", end='', flush=True)
    # Perform the SVD
    U, s, VT = np.linalg.svd(states[idx,:].T, full_matrices=False)
    # Add the new high-dimensional point
    C.append(s)
C = np.array(C)
print('\nDone...')

# Create the kernel with the KDE
r=5 # Number of components (singular values or dimensions of the latent space) used
values = np.stack(C[:,0:r])

from sklearn.neighbors import KernelDensity
kernel = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(values)


#####################################################################
# STEP 2: TEST THE METHOD
#####################################################################

print('Runnig all signals on the ESN model...')
# Get all inputs in vector X
X = df[features].values
# And desired output in vector Y
Y = df['activity_label'].values
states = reservoir.run(X)
Y_out = readout.run(states)

# Clip the output for representation purposes
# both class 0 and 12 are invalid
Y_out = np.clip(Y_out, 0, 12)

# Apply the sliding window algorithm
C = []
Q = X.shape[0]
t = np.arange(Q).reshape(-1,1)*tm
S = 20
L = 140 

print(f'Decomposing (wnd siz {L}, stride {S})...')
rango = np.arange(0,Q-L,S)
for i in rango:
    idx = np.arange(i,i+L)
    print(f"\rWindow {i} of {rango[-1]}", end='', flush=True)
    U, s, VT = np.linalg.svd(states[idx,:].T, full_matrices=False)
    C.append(s)
C = np.array(C)
print('\nDone...')


# Get the scores
logprobX = kernel.score_samples(C[:,0:r])

# Adjust lengths
t_adj = t[:Y_out.shape[0]]
Y_adj = Y[:Y_out.shape[0]]

# Plot results
plt.ion()
plt.figure()
plt.subplot(2,1,1)
plt.plot(t,X[:,0:3]/np.max(np.abs(X[:,0:3]))/3)
plt.plot(t,X[:,3:6]/np.max(np.abs(X[:,3:6]))/3+1)
plt.plot(t,X[:,6:9]/np.max(np.abs(X[:,6:9]))/3+2)
plt.grid()
ax=plt.gca()
ax.set_xlim(0,t_adj[-1])
ax.set_yticks([0, 1, 2])
ax.yaxis.set_ticklabels(["IMU1", "IMU2", "IMU3"])
plt.yticks(rotation=90)

plt.title('IMU Signals')
plt.ylabel('acceleration (normalized)')


cc = np.array([1 if i>=-10 else 0 for i in logprobX])
cc_exp = np.kron(cc,np.ones(S))

plt.subplot(2,1,2)
plt.plot(t_adj,Y_adj,label='Real')
plt.plot(t_adj,Y_out,color='black',label='Estimated',alpha=0.3)
idx=np.where(cc_exp==0)
plt.scatter(t_adj[idx],Y_out[idx],c='red',marker='s', s=2)
idx=np.where(cc_exp==1)
plt.scatter(t_adj[idx],Y_out[idx],c='green',marker='s', s=20)
plt.grid()
ax=plt.gca()
ax.set_xlim(0,t_adj[-1])
ax.set_ylim(0,12)

plt.title('Estimated activity class')
plt.xlabel('time (s)')
plt.ylabel('activity class')