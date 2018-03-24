
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import sklearn
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt


# In[4]:


data = pd.read_csv('PwrData_2018-3-19-20-16-24.csv')
data = data.iloc[0:113239,:]


# In[5]:


data.head(10)


# In[6]:


data.tail(10)


# In[7]:


data.columns


# In[8]:


X = data[['Elapsed Time (sec)',' CPU Utilization(%)']]
y = data['Processor Power_0(Watt)']


# In[10]:


lr = LinearRegression()
y_lin = lr.fit(X,y).predict(X)
error_linear_regression = np.sqrt(((y_lin-y)**2).mean())


# In[11]:


error_linear_regression


# In[13]:


lr = LinearRegression()
y_pred = cross_val_predict(lr,X,y, cv = 5)
error_linear_regression_cv = np.sqrt(((y_pred-y)**2).mean())


# In[14]:


error_linear_regression_cv


# In[15]:


#Cross validation inutile pour la regression linéaire ce qui était prévisible


# In[ ]:


clf = SVR(C = 1.0, epsilon = 0.2)
y_rbf = clf.fit(X,y).predict(X)
error_linear_SVR = np.sqrt(((y_rbf-y)**2).mean())

