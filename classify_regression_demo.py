# !/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report

# 逻辑回归
def logic_regression():
    """
        公式：
                                1
            hθ = g(θ^Tx) = ————————————
                           1 + e^(-θ^Tx)
                       1
            g(z) = ——————————
                   1 + e^(-z)
            输入：[0,1]区间的概率，默认值0.5作为阈值
            g(z)：sigmoid函数，z为回归结果
        损失函数：
            与线性回归原理相同，但是由于是分类问题。损失函数不一样。
            只能通过梯度下降求解。
            对数似然损失函数：
                                 { -log(hθ(x))     if y = 1
                cost(hθ(x), y) = {
                                 { -log(1 - hθ(x)) if y = 0
                hθ(x)为x的概率值
                说明：在均方误差中不存在多个最低点，但是对数似然损失函数，会存在多个低点的情况
        完整的损失函数：
                             m
            cost(hθ(x), y) = ∑-yilog(hθ(x)) - (1 - yi)log(1 - hθ(x))
                            i=1
            cost损失值越小，那么预测的类别精准度更高
    """

    """
        penalty:正则化方式默认值l2，
        C为回归系数默认值1.0
    """
    # 1、原始数据
    # 地址：https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
    # 数据：https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data
    column_names = ["Sample code number",
                    "Clump Thickness",
                    "Uniformity of Cell Size",
                    "Uniformity of Cell Shape",
                    "Marginal Adhesion",
                    "Single Epithelial Cell Size",
                    "Bare Nuclei",
                    "Bland Chromatin",
                    "Normal Nucleoli",
                    "Mitoses",
                    "Class"]
    data = pandas.read_csv("classify_regression/breast-cancer-wisconsin.data", names=column_names)
    # print(data)

    # 2、数据处理
    # 缺失值处理
    data = data.replace(to_replace="?", value=numpy.NAN)
    # 删除缺失值数据
    data = data.dropna()
    # 特征值，目标值
    x = data[column_names[1:10]]
    y = data[column_names[10]]
    # 数据分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 3、特征工程
    std = StandardScaler()
    x_train = std.fit_transform(x_train, y_train)
    x_test = std.transform(x_test)

    # 4、算法工程
    lr = LogisticRegression(penalty="l2", C=1.0)
    # 训练
    lr.fit(x_train, y_train)
    print("权重值：", lr.coef_)

    # 5、评估
    print("准确率：", lr.score(x_test, y_test))
    y_predict = lr.predict(x_test)
    print("召回率：", classification_report(y_test, y_predict, labels=[2, 4], target_names=["良性", "恶性"]))
    print("均方误差：", mean_squared_error(y_test, y_predict))

if __name__ == '__main__':
    logic_regression()