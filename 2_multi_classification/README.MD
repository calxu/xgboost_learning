# 多分类任务

XGBoost在处理多分类问题时也是在树模型的基础上进行转换的，二分类是基于sigmoid函数进行变化，多分类是基于softmax函数进行变换。相关代码可参考当前文件夹的ipynb文件。



## 主要参数

XGBoost中解决多分类问题的主要参数如下：

1. objective：该参数可指定为 multi:softmax 和 multi:softprob ，两者均是指定学习任务为多分类。multi:softmax 和 multi:softprob 都是通过 softmax 函数解决多分类问题的。其中 multi:softprob 是输出一个 ndata * nclass 向量，表示样本属于每个分类的预测概率；而multi:softmax 是输出得最终结果，即取每个分类的预测概率的最大值作为其最终类别。**multi:softmax 和 multi:softprob 的区别可参考代码。**
2. num_class：说明在该分类任务的类别数量，**注：该参数不同于二分类问题，分类类别的数目必须设置，而二分类不需要指定。**
3. eval_metric：多分类评估函数主要有 merror 和 mlogloss。



## 注意事项

1. 训练集文件和测试集文件分别为 seeds_dataset.txt.train 和 seeds_dataset.txt.test，数据集中为7个特征列，最后1列为label列，类别label编号是从0开始计数。
2. 最终训练完成后树的棵为 num_round * num_class （即迭代轮数与类别数目的乘积），**它与二分类任务不同，二分类的迭代轮数即为最终树的棵数。** 





## 参考文献

1. 《深入理解XGBoost》，何龙
2. [XGBoost参数说明](https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters)，XGBoost官方


