# 模型保存与解析

这里主要介绍下XGBoost模型的训练、保存与解析，给出了一个完整案例。数据是依赖的 XGBoost 官方的文件，详见[链接](https://github.com/dmlc/xgboost/tree/master/demo/CLI/binary_classification) 。对原始数据处理成 LibSVM 的数据格式（即 agaricus.txt.train文件 和 agaricus.txt.test文件），映射文件（featmap.txt文件）也是基于原始特征进行加工而来。**注意libSVM数据文件索引是从1开始，featmap.txt文件索引是从0开始。**



## LibSVM数据格式

LibSVM是机器学习中经常采用的一种数据格式。

格式如下：<label> <index1>:<value1> <index2>:<value2> ...

label为训练数据集的目标值；index为特征索引（**从1开始**）；value是该特征的取值，如果某一特征的值缺省，则该特征可以空着不填。

我们这里的 agaricus.txt.train 和 agaricus.txt.test 即为训练和测试文件。



## 映射文件说明

映射文件是完成索引和特征名称的映射，代码里的文件是 featmap.txt 文件。

格式如下：<featureid> <featurename> <q or i or int>\n

其中：featureid为特征索引（**从0开始**）； featurename为特征名称； 

q or i or int 为特征的数据类型，其中 q 代表征是一个连续值，如距离、价格等；i代表特征是一个二值特征（即特征只有两个取值），一般为0或1；int代表特征是整型值；

我们这里 featmap.txt 文件中的所有特征都是二值特征，因为该数据集中大部分是离散型的类别特征，因此经 one-hot 编码处理后都是二值特征，所以文件中最后一列标识列都为i。



## 注意事项

1. 模型训练完后，可通过 save_model 方法将模型保存成二进制模型文件，该文件不可用 vim 打开可读。而 dump_model 是输出JSON文本文件的格式，不同于 save_model 方法， 可用 vim 打开可读，文件可用来做可视化、可解释性方面的工作。

2. LibSVM数据格式的特征索引是从1开始计数；而映射文件的特征索引是从0开始计数。



## 参考文献

1. 《深度理解XGBoost》，何龙
