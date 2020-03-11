import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn import linear_model
data=pd.read_csv("GPU.csv",index_col=0,usecols=range(8))

values=data.values
wide,len=values.shape
reg=linear_model.LinearRegression()
##################Standardization#################
values[:,range(len-1)]=values[:,range(len-1)]/values[0,range(len-1)]
values[:,len-1]=values[:,len-1]/values[0,len-1]
#################Linear Regression################
values=values[1:,:]
List=list(range(wide-1))
selected=random.sample(List,k=20)
un_selected=list(set(List).difference(set(selected)))

X_train=values[selected,:-1]
y_train=values[selected,len-1]
reg.fit(X_train,y_train)
#####################Prediction#####################
X_test=values[un_selected,:-1]
y_test=values[un_selected,len-1]
#y_predict=data.values[0,len-1]/reg.predict(X_test)
y_predict=reg.predict(X_test)*data.values[0,len-1]
print('Prediction: ', y_predict)
print('Regression Score: ', reg.score(X_test,y_test))
print('Coefficients: ',reg.coef_)
print('intercept: ',reg.intercept_)






