#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 导入常用包
import xgboost as xgb
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd


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

num_round = 50


# In[7]:


watchlist = [(xgb_test, 'eval'), (xgb_train, 'train')]


# In[8]:


bst = xgb.train(params, xgb_train, num_round, watchlist) 


# In[9]:


# get_fscore采用默认的weight指标计算特征重要性

bst.get_score(importance_type="weight")
# bst.get_fscore()

# 降序显示
# importance = bst.get_score(importance_type="weight")
# importance = sorted(importance.items(), key = lambda x: x[1], reverse = True)
# importance


# In[10]:


# total_gain = gain * weight
# bst.get_score(importance_type = "gain")
bst.get_score(importance_type = "total_gain")


# In[11]:


# bst.get_score(importance_type = "cover")
bst.get_score(importance_type = "total_cover")


# In[12]:


# 特征重要性可视化
import matplotlib.pyplot as plt

xgb.plot_importance(bst, height = 0.5)
plt.show()

