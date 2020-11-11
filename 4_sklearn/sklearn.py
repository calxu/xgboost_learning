#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 导入常用包
import xgboost as xgb

import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.metrics import roc_auc_score


# # 二分类，乳腺癌数据集

# In[2]:


cancer = datasets.load_breast_cancer()


# In[3]:


X = cancer.data
Y = cancer.target


# In[4]:


# X.shape; Y.shape


# In[5]:


kf = KFold(n_splits = 3, shuffle = True)


# In[6]:


# X[:10]
# Y[:10]


# In[7]:


i = 0

for train_idx, test_idx in kf.split(X):
    model = xgb.XGBClassifier().fit(X[train_idx], Y[train_idx])
    preds = model.predict(X[test_idx])
    labels = Y[test_idx]
    
    print(i, roc_auc_score(labels, preds))
    i += 1


# # 回归，波士顿房价数据集

# In[8]:


boston = datasets.load_boston()


# In[9]:


X = boston.data
Y = boston.target


# In[10]:


# X.shape, Y.shape


# In[11]:


kf = KFold(n_splits = 3, shuffle = True)


# In[12]:


# X[:10]
# Y[:10]


# In[13]:


i = 0

for train_idx, test_idx in kf.split(X):
    model  = xgb.XGBRegressor().fit(X[train_idx], Y[train_idx])
    preds  = model.predict(X[test_idx])
    labels = Y[test_idx]
    
    print(i, mean_squared_error(labels, preds))
    i += 1

