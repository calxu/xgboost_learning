# 单个样本的归因分析

特征重要性 gain、total_gain、weight、total_weight 和 cover 这些指标都是整体评估，无法实现单个样本的个性化评估。无法了解各特征对单个样本预测结果的贡献，如果需要了解各特征对单个样本的权重则需要个性化归因方法，即 Saabas 和 SHAP 方法。

其中 SHAP 方法较为常用，关键的代码如下，关于 predict 方法函数可参考官方 [链接](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.predict) 。

```
bst.predict(xgb_test, pred_contribs = True)
```

Sabbas 方法相对实现简单、容易理解，**但可能出现更重要的特征会分配较低的重要性，所以不常用。** 关于 Sabbas 方法的调用如下，即加上 approx_contribs 参数即可。

```
bst.predict(xgb_test, pred_contribs = True, approx_contribs = True)
```



关于 SHAP 和 Sabbas 方法的理论可参考 [知乎文章](https://zhuanlan.zhihu.com/p/186204351) 。**在 SHAP 和 Sabbas 方法之间，我们更多地是应用 SHAP 方法进行个性化分析。但在工业界一般很少用 SHAP 单独分析这样做，更多的是圈一批样本，看这批样本的整体情况，样本整体情况一般还是依据特征重要性，其中更多的是看 total_gain 指标。** 



## 参考文献

1. [XGBoost官方文档 predict 方法](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.predict) ，XGBoost官方
2. [模型可解释性(2)-SHAP计算过程](https://zhuanlan.zhihu.com/p/186204351) ，知乎
3. 《深入理解XGBoost》，何龙
