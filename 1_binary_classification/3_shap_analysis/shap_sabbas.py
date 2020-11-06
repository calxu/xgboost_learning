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
          "eta": 1,
          "max_depth": 2
         }

num_round = 10


# In[7]:


watchlist = [(xgb_test, 'eval'), (xgb_train, 'train')]


# In[8]:


bst = xgb.train(params, xgb_train, num_round, watchlist)


# In[9]:


# SHAP预测样本归因分析

pred_contribs = bst.predict(xgb_test, pred_contribs = True)

# 打印第一个样本：因为该测试集包含30个特征，因此输出向量是31维，最后一列即为偏置项
print(pred_contribs[0])


# In[10]:


# Sabbas预测样本归因分析

pred_contribs = bst.predict(xgb_test, pred_contribs = True, approx_contribs = True)

# 输出结果同上，30个特征对当前样本的贡献 和 最后一列偏置项
print(pred_contribs[0])

