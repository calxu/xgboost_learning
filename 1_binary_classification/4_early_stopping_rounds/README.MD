# 提前停止迭代 eary_stopping_rounds 参数

在XGBoost中，通过设置参数 early_stopping_rounds ，实现在一定迭代次数内评估指标没有 **任何一次提升** ，就停止训练。在设置 early_stopping_rounds 之前，必须要设置 xgboost.train 方法中的 evals 参数，该参数是一个 list 列表，当 evals 参数列表中存在多个值时，取最后一个值，**这块可进一步参考代码示例** 。

对于 xgboost.train 方法的最常用的参数如下（**这里给出最常用的**），其它详细的参数细节可参考官方 [链接](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.train) 。

```
xgboost.train(params, dtrain, num_boost_round = 10, evals = (), early_stopping_rounds = None)
```

其中 params 为字典类型，它是基分类器的参数描述；dtrain 为 DMatrix 数据类型的训练集数据；num_boost_round为迭代的次数；evals 是观察的列表类型（列表中至少一个值），例 [(xgb_train, 'train')] ，在每一轮训练阶段帮我们追踪模型的迭代效果，也是提前停止迭代的依据指标（如果evals 列表有多个值，取最后一个值）；early_stopping_rounds 为提前停止迭代的参数，也是本次主要实践的内容。



模型返回的是最后一轮迭代次数，而非最好的一次迭代，**这块可进一步参考代码示例** 。如果在训练过程中提前停止迭代发生了，则训练出的模型会有三个属性值：bst.best_score，bst.best_iteration 和 bst.best_ntree_limit ，其中 bst.best_score 为模型训练中评估指标的最优值，bst.best_iteration 为模型最优迭代轮数，bst.best_ntree_limit 为最优的前n棵树。



## 参考文献

1. [XGBoost 的 train 方法](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.train) ，XGBoost官方

2. 《深入理解XGBoost》，何龙


