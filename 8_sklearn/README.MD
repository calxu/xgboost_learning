# XGBoost的sklearn接口使用

XGBoost有一个 sklearn 接口版本，这里展示了如何通过 XGBoost 的 sklearn 接口版本解决分类、回归问题。sklearn 接口版本通过 XGBClassifier 解决分类问题，通过 XGBRegressor 解决回归问题。与平时 sklearn 使用习惯保持一致，我们可以通过 fit 方法完成模型拟合。

详细代码可参考该目录下示例代码，其中关键代码如下：

```
xgb.XGBClassifier().fit(X, Y)
xgb.XGBRegressor().fit(X, Y)
```

**个人理解：二分类问题相对评估结果会稳定很多，回归的输出结果相对波动会比较大。**



# 参考文献

1. [XGBoost 的 sklearn API](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn) ，XGBoost官方
2. 《深入理解XGBoost》，何龙