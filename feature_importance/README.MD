# XGBoost输出特征重要性

从模型层面来讲，特征的重要性表明每个特征在模型中的价值；从业务层面来讲，对于重要的特征，是重点迭代关注的对象（工业界经常底层其它特征因为各种原因都不更新了，但重要性排在前面的特征一定要保证正常迭代更新）。

特征重要性的衡量有不同维度的视角，不同维度的视角得出的特征重要性会有不同的结果。
XGBoost主要有3个衡量特征重要性的指标 weight 、gain 和 cover，**其中最常用得是 weight 和 gain** 。
**weight 表示在模型中一个特征被选作分裂特征的次数；gain 表示特征在模型中被使用的平均增益（增益通过损失函数的变化度量，即 
$Gain = \frac{1}{2} [ \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} ] - \gamma$ ）
![1](http://latex.codecogs.com/svg.latex?\int_a^bf(x)\ dx)

注意这里是平均增益，即该特征增益总和除以该特征出现次数；** cover 并不常用，它表示特征在模型中被使用的平均覆盖率（通过节点的二阶梯度和来计算的），当损失函数为平方损失（即默认参数），
那么该参数的含义就对应指定结点训练样本的数目，官方对cover的描述稍微有点抽象，可以看下 [博客](https://blog.csdn.net/sujinhehehe/article/details/84201415#commentBox) 举的一个例子（例子中是平方损失即为分裂训练样本的个数）。

后续XGBoost增加了 total_gain 和 total_cover，其中 total_gain = gain * weight 、total\_cover = cover * weight 。**total_gain 是非常常用的评估特征重要性的指标，因为 total_gain 是特征出现次数和特征的增益的综合考虑。**

关于XGBoost的 weight、gain、cover、total_gain 和 total_cover 的源码讲解实现可以参考 [链接](https://zhuanlan.zhihu.com/p/64759172) 。



XGBoost提供了获取特征重要性的方法函数 get_score 和 get_fscore 。其中 get_fscore 函数采用了默认的 weight 指标计算特征重要性评分，该函数不常用；**而 get_score 可通过 importance_type 进行参数选择 weight、gain、cover、total_gain 或 total_cover 。** 下面的引用为该方法的调用说明，该方法详细说明可参考官方 [链接](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.get_score) 。**其中 total_gain 最为常用。**

> `get_score`(*fmap**=**''*, *importance_type**=**'weight'*)
>
> Get feature importance of each feature. Importance type can be defined as:
>
> - ‘weight’: the number of times a feature is used to split the data across all trees.
> - ‘gain’: the average gain across all splits the feature is used in.
> - ‘cover’: the average coverage across all splits the feature is used in.
> - ‘total_gain’: the total gain across all splits the feature is used in.
> - ‘total_cover’: the total coverage across all splits the feature is used in.



XGBoost内置了一种可视化展示特征重要性排序的方法，即 plot_importance，该函数将特征重要性评分降度排列，使用户可以直观了解重要特征的分布情况，该方法的详细说明可参考官方 [链接](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.plot_importance) 。



特征重要性的代码参考 github。



## 参考文献

1. [xgboost特征重要性指标：weight，gain，cover](https://blog.csdn.net/sujinhehehe/article/details/84201415#commentBox) ，CSDN
2. [XGBoost的 get_fscore 和 get_score 方法函数](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.get_fscore)，XGBoost官方
3. [机器学习的特征重要性究竟是怎么算的](https://zhuanlan.zhihu.com/p/64759172) ，知乎


