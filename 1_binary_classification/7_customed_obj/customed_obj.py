#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 导入常用包
import xgboost as xgb
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel


# In[2]:


# 数据集
cancer = datasets.load_breast_cancer()
X = cancer.data
Y = cancer.target


# In[3]:


# 数据集的情况
# X.shape
# Y.shape
# X, Y


# In[4]:


# 拆分训练集、测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/5., random_state = 8)


# In[5]:


xgb_train = xgb.DMatrix(X_train, label = Y_train)
xgb_test  = xgb.DMatrix(X_test,  label = Y_test)


# In[6]:


params = {
    "objective": "binary:logistic",
    "booster": "gbtree",
    "eta": 0.1,
    "max_depth": 2
}


# In[7]:


num_round = 30


# In[8]:


watchlist = [(xgb_test, 'eval'), (xgb_train, 'train')]


# In[9]:


bst1 = xgb.train(params, xgb_train, num_round, watchlist)


# In[10]:


def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0 - preds)
    return grad, hess


# In[11]:


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    return 'error', float(sum(labels != (preds > 0.5))) / len(labels)


# In[12]:


params = {
    'objective': 'reg:logistic',
    "booster": "gbtree",
    "eta": 0.1,
    "max_depth": 2
}


# In[13]:


bst2 = xgb.train(params, xgb_train, num_round, watchlist, obj = logregobj, feval = evalerror)


# In[14]:


bst1.predict(xgb_test)#, output_margin = True)


# In[15]:


bst2.predict(xgb_test)

