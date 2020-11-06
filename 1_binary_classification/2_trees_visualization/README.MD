# XGBoost模型可视化

XGBoost是决策树集合的模型，可视化决策树可以更加直观并且有利于模型的分析。XGBoost模型为我们提供了两个可视化决策树的方法函数 plot_tree 和 to_graphviz ，plot_tree方法函数有点不太好使（可视化显示图片太小，该方法依赖于graphviz包），我这里使用 to_graphviz 方法可以得到清晰的可视化，该方法函数官方使用文档可参考 [链接](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.to_graphviz) 。

两种方法都需要依赖 graphviz 包，centos系统可以通过 `yum install graphviz`  来安装 graphviz 包。to_graphviz 方法使用关键代码如下：

```
# xgb单棵树可视化，num_tree共50棵树，取值为 [0, 49]
xgb.to_graphviz(bst, fmap = '', num_trees = 2)
```

bst 为训练的booster模型，num_trees为要可视化显示的第几棵树，详细代码可参考 github 里代码。



## 参考文献

1. [XGBoost的 to_graphviz 方法函数](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.to_graphviz) ， XGBoost官方

