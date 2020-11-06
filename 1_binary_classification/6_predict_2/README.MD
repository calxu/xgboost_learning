# predict 未经（sigmoid）转化的预测值

此处的预测值为未经行转化（如sigmoid、softmax等）的原始值。关于 predict 方法函数详情可参考 [链接](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.predict) ，其中关键代码如下。

```
bst.predict(xgb_test, output_margin = True)
```

其中 output_margin 参数是表示是否经过 sigmoid 转化。



# 参考文献

1. [XGBoost 的 predict 方法](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.predict) ，XGBoost官方

2. 《深入理解XGBoost》，何龙

