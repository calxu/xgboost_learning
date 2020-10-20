# 通过前n棵树进行预测

默认情况下，XGBoost会使用模型中所有决策树进行预测。在某些场景下，可以通过指定参数 ntree_limit 只使用前 n 棵树进行预测，比如：early_stopping_rounds 提前停止迭代发生时，则可以用 ntree_limit 取最优的前 n 棵树进行预测。

关键代码如下，predict 方法其它详细的参数细节可参考官方 [链接](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.predict) 。

```
bst.predict(xgb_test, ntree_limit = 10)
```

其中 xgb_test 为 DMatrix 数据类型的训练集数据；ntree_limit 为指定前 n 棵树进行预测，该参数默认为0（即取所有树进行预测）。详细可参考代码示例。



# 预测叶子结点索引

在XGBoost的树模中，每个叶子节点在其所在的决策树中都有一个唯一索引。在预测阶段，样本在每棵决策树中都会被划分到唯一的叶子节点，所有叶子节点索引组成一个向量，在某种程谨防上，该向量可以代表样本在模型中被划分的情况。

XGBoost支持用户获得这样的叶子节点索引向量，只需将参数 pred_leaf 置为 True 即可。

关键代码如下，predict 方法其它详细的参数细节可参考官方 [链接](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.predict) 。

```
bst.predict(xgb_test, ntree_limit = 10, pred_leaf = False)
```

其中 pred_leaf 参数是否输出叶子结点索引，当其为 True 时，输出其索引；当其为 False 时，就是正常输出的概率分。



# 参考文献

1. [XGBoost 的 predict 方法](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.predict) ，XGBoost官方

2. 《深度理解XGBoost》，何龙
