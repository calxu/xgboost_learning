#  XGBoost自定义目标函数和评估函数

虽然XGBoost提供了自定义目标函数和评估函数，但感觉自定义目标函数接口很不友好。

> 自定义目标函数时需返回其一阶、二阶梯度，因此目标函数需满足二次可微，通过自定义目标函数得到的预测值是模型预测的原始值，不会进行任何转换（如 sigmoid 转换、softmax 转换）。自定义 objective 之后，模型并不知道该任务是什么类型的任务，因此也就不会再做转换。自定义目标函数后，XGBoost 内置的评估函数不一定适用。比如用户自定义了一个 logloss 的目标函数，得到的预测值是没有经过 sigmoid 转换的，而内置的评估函数默认是经过转换的，因此评估时就会出错。所以一般目标函数和评估函数都需要同时自定义。

自定义目标函数的 Demo为 ./demo/guide-python/custom_objective.py 文件，实验测试代码可参考当前目录下 github 的实验代码，其中关键代码如下：

```
def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0 - preds)
    return grad, hess
    
def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    return 'error', float(sum(labels != (preds > 0.5))) / len(labels)
    
params = {
    'objective': 'reg:logistic',
    "booster": "gbtree",
    "eta": 0.1,
    "max_depth": 2
}

xgb.train(params, xgb_train, num_round, watchlist, obj = logregobj, feval = evalerror)
```



**个人理解：XGBoost 对自定义目标函数接口非常不友好，objective中必须指定，若不指定则默认为 "reg:linear"（即损失函数为均方误差  
<img src="./images/reg_linear.svg" alt="avatar" style="zoom:20%;" />  ），train方法中obj参数仅支持定义损失函数的一、二阶偏导，无法定义损失函数本身，损失函数的定义只能修改XGBoost源码。** 

其中objective目标函数中的 "reg:logistic" 和 "binary:logistic" 区别主要在于前者默认评价指标为 error，后者默认评价指标为 rmse（具体实验验证可看下这篇 [CSDN文章](https://blog.csdn.net/phyllisyuell/article/details/81005509) ）。在XGBoost中通常在对对数似然取负数（可参考源码：./doc/tutorials/model.rst 的50行；./R-package/tests/testthat/test_callbacks.R 的252行），形式如下：


<img src="./images/obj.svg" alt="avatar" style="zoom:110%;" align="left" />

 

其中 <img src="./images/pred.svg" alt="avatar" style="zoom:110%;" /> 为上一轮迭代模型的预测值（即 pred，也是sigmoid函数）。


对上式求一阶导为：<img src="./images/grad.svg" alt="avatar" style="zoom:110%;" />


二阶导为：<img src="./images/hess.svg" alt="avatar" style="zoom:110%;" />


其中 y 为样本的label，<img src="./images/y.svg" alt="avatar" style="zoom:110%;" /> 为上一次模型的预测值（未进行 sigmoid 转换）。根据上述公式，loggreobj 函数实现了二分类对数似然函数一阶梯度和二阶梯度的计算，并作为函数返回值返回，**这里只返回目标函数的一阶和二阶导数，目标函数本身无法定义，只有通过修改源码，所以我感觉这是XGBoost的缺陷**。evalerror 函数实现了自定义评估函数。这里注意因为模型产出的预测值 preds （即<img src="./images/y.svg" alt="avatar" style="zoom:110%;" />）为原始值，所以在自定义目标函数内部对 preds （即<img src="./images/y.svg" alt="avatar" style="zoom:110%;" />）进行了 sigmoid 转换，转换之后再计算一阶、二阶梯度。

在自定义评估函数中 preds （即 <img src="./images/y.svg" alt="avatar" style="zoom:110%;" />） 也为转换前的原始值，因此也需要转换，转换后以 0.5 为阈值，大于 0.5 为正例，小于等于 0.5 为负例。

在实现自定义函数后，指定相关参数 obj 和 feval 。这两个参数均为函数引用，默认值为 None。



# 参考文献

1. [XGBoost 的 train 方法](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.train) ，XGBoost官方
2. 《深入理解XGBoost》，何龙
3. [xgboost 逻辑回归：objective参数（reg:logistic,binary:logistic,binary:logitraw,）对比分析](https://blog.csdn.net/phyllisyuell/article/details/81005509) ，CSDN博客


