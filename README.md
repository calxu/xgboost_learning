# xgboost_learning

XGBoost目前在工业界非常常用，所以有必要把XGB的原理和应用系统复习一遍。
这里整理一些XGBoost学习笔记。



## 文件说明

1. 1_feature_importance: 特征重要性的说明与应用（gain、total_gain、cover、total_cover、weight）

2. 2_trees_visualization: 单棵树可视化分析

3. 3_shap_analysis: 单个样本的归因分析（SHAP 和 Saabas）

4. 4_early_stopping_rounds: 提前停止迭代 eary_stopping_rounds 参数

5. 5_predict_1: predict 方法函数（通过前n棵树进行预测、预测叶子结点索引）

6. 6_predict_2: predict 方法函数（predict 未经 sigmoid 转化的预测值）

7. 7_customed_obj: XGBoost自定义目标函数和评估函数
