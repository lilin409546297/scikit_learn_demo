# !/usr/bin/python
# -*- coding: UTF-8 -*-
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib


def k_near_save_load():
    # 1、原始数据
    li = load_iris()
    # 2、处理数据
    data = li.data
    target = li.target
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25)
    # 3、特征工程
    std = StandardScaler()
    x_train = std.fit_transform(x_train, y_train)
    x_test = std.transform(x_test)
    # 4、算法
    knn_gc = KNeighborsClassifier()
    # 构造值进行搜索
    param= {"n_neighbors": [2, 3, 5]}
    # 网格搜索
    gc = GridSearchCV(knn_gc, param_grid=param,cv=4)
    gc.fit(x_train, y_train)

    # 5、评估
    print("测试集的准确率：", gc.score(x_test, y_test))
    print("交叉验证当中最好的结果：", gc.best_score_)
    print("选择最好的模型：", gc.best_estimator_)
    print("每个超参数每次交叉验证结果：", gc.cv_results_)

    # 6、保存模型
    joblib.dump(gc, "model/k_near.pkl")

    # 7、模型的加载使用
    model = joblib.load("model/k_near.pkl")
    m_predict = model.predict(x_test)
    print("保存模型的测试结果：", m_predict)

if __name__ == '__main__':
    k_near_save_load()