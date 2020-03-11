import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn import tree
from sklearn.tree import export_graphviz
runtime_data=pd.read_csv("runtime_data.csv",index_col=0,usecols=range(38))
GPU_spec=pd.read_csv("GPU.csv",index_col=0,usecols=range(7))
GPU_spec_values=GPU_spec.values/GPU_spec.values[0,:]
Batch_size=runtime_data.values[0,:]
Input_size=runtime_data.values[1,:]
CPU_values=runtime_data.values[2,:]
GPU_values=runtime_data.values[2,:]/runtime_data.values[3:,:]
#GPU_values=runtime_data.values[3:,:]/runtime_data.values[3,:]
wide,len=GPU_values.shape
GPU_inf=GPU_values[:,range(0,len,2)]
GPU_train=GPU_values[:,range(1,len,2)]
############################################
data_inf=np.array(9*[0])
data_train=np.array(9*[0])
wide_inf,len_inf=GPU_inf.shape
wide_train,len_train=GPU_train.shape
for i in range(wide):
    spec=GPU_spec_values[i,:]
    for j in range(len_inf):
        s=np.append(spec,Batch_size[2*j])
        s=np.append(s,Input_size[2*j] )
        s=np.append(s,GPU_inf[i,j])
        data_inf=np.vstack((data_inf,s))
    for j in range(len_train):
        s = np.append(spec, Batch_size[2 * j+1])
        s = np.append(s, Input_size[2 * j+1])
        s = np.append(s, GPU_train[i, j])
        data_train = np.vstack((data_train, s))
data_inf=data_inf[1:,:]
data_train=data_train[1:,:]
###############################################
reg_inf=tree.DecisionTreeRegressor()
wide,len=data_inf.shape
List=list(range(wide))
selected=random.sample(List,k=wide-10)
un_selected=list(set(List).difference(set(selected)))
X_train=data_inf[selected,:-1]
y_train=data_inf[selected,-1]
X_test=data_inf[un_selected,:-1]
y_test=data_inf[un_selected,-1]
reg_inf.fit(X_train,y_train)
y_predict=reg_inf.predict(X_test)
print(y_test)
print('Prediction: ', y_predict)
print('Regression Score: ', reg_inf.score(X_test,y_test))
print('Feature_importance: ',reg_inf.feature_importances_)


