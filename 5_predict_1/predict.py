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
          "max_depth": 3,
          "eval_metric": "auc"
         }


# In[7]:


watchlist = [(xgb_test, 'eval'), (xgb_train, 'train')]


# In[8]:


bst = xgb.train(params, xgb_train, num_boost_round = 20, evals = watchlist)


# # 通过前n棵树进行预测 和 通过全部树进行预测 的比较

# In[9]:


# 使用前n棵树进行预测
pred_1 = bst.predict(xgb_test, ntree_limit = 10)

# 计算前10棵树预测的AUC
roc_auc_score(Y_test, pred_1)


# In[10]:


# 使用所有决策树进行预测
pred_2 = bst.predict(xgb_test)

# 计算所有决策树进行预测的AUC。最终预测出来的两者AUC不同
roc_auc_score(Y_test, pred_2)


# # 预测叶子节点索引

# In[11]:


# 通过前10棵树预测
# 叶子节点索引列数 = 预测树的棵数

leaf_index = bst.predict(xgb_test, ntree_limit = 10, pred_leaf = True)
print(leaf_index.shape)
print(leaf_index)


# In[12]:


# 通过整个模型预测（20棵树预测）
# 叶子节点索引列数 = 预测树的棵数

leaf_index = bst.predict(xgb_test, pred_leaf = True)
print(leaf_index.shape)
print(leaf_index)

