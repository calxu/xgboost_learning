# 回归问题

XGBoost本身采用的就是回归树，所以它可以解决回归问题。注：解决二分类或多分类问题，也是在回归树结果的基础上加上 sigmoid 或 softmax 来分别解决二分类 或 多分类问题的。而回归问题最终无需转换。



## 主要参数

### 目标函数

1. reg:linear：线性回归，用于数据符合正态分布的回归问题，目标函数为最小二乘。基分类器为树模型，而并非线性模型（线性模型由booster参数指定）。
2. reg:logistic：逻回回归，目标函数与二分类目标函数一致，区别是不再进行sigmoid转换，默认评价指标为rmse。
3. count:poisson：计数数据的泊松回归。
4. reg:gamma：对数连接函数下的伽马回归。
5. reg:tweedie：对数连接函数下的 tweedie 回归。



### 评估函数

评估函数主要有 RMSE 和 MAE，另外还有 poisson-nloglik、gammanloglik、gamma-deviance、tweedie-nloglik等用于特定回归的评估指标。其中 RMSE（Root Mean Square Error，均方根误差）是回归模型中最常采用的评估指标之一，是预测值与真实值偏差的平方和与样本数比值的平方根，也是XGBoost默认采用的评估指标。





## 参考文献

1. 《深入理解XGBoost》，何龙
