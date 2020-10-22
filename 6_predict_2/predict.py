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


params = {"objective": "binary:logistic",
          "booster": "gbtree",
          "eta": 0.1,
          "max_depth": 5
         }


# In[7]:


num_round = 50


# In[8]:


watchlist = [(xgb_test, 'eval'), (xgb_train, 'train')]


# In[9]:


bst = xgb.train(params, xgb_train, num_boost_round = 20, evals = watchlist)


# In[10]:


# output_margin 参数设为 True，表示最终输出的预测值为未进行 sigmoid 转化的原始值
pred_test = bst.predict(xgb_test, output_margin = True)
pred_test_sigmoid = bst.predict(xgb_test)


# In[11]:


# 将 原始值进行 sigmoid 转化
1.0 / (1.0 + np.exp(-pred_test[:30]))


# In[12]:


# 和原始值进行 simoid 转化的值进行逐一比较
pred_test_sigmoid[:30]

