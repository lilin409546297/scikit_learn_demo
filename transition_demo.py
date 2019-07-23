# !/usr/bin/python
# -*- coding: UTF-8 -*-
import jieba
import numpy
from sklearn.datasets import load_iris, load_boston
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Imputer

# 字典特征提取
def dict_data():
    # sparse=False：one-hot, True:矩阵
    dict = DictVectorizer(sparse=True)
    data = dict.fit_transform([{"city": "四川", "temperature": 20}, {"city": "北京", "temperature": 30}])
    # 转换成矩阵
    print(data.toarray())
    # 特征名称
    print(dict.get_feature_names())
    # 逆向转换成字典
    print(dict.inverse_transform(X=data))

# 文本特征提取
def count_data():
    cv = CountVectorizer()
    # data = cv.fit_transform(["I love you", "I like you"])
    # data = cv.fit_transform(["人生 苦短 我 喜欢 Python", "人生 漫长 我 讨厌 Python"])
    # 中文文字分词
    data = cv.fit_transform([' '.join(jieba.cut("人生苦短，我喜欢Python")), ' '.join(jieba.cut("人生漫长，我讨厌Python"))])
    print(data)
    print(cv.get_feature_names())
    print(data.toarray())

# term frequency(词出现的频率) and inverse document frequency(log(总文档数量/改词频率))
def tf_idf_data():
    cv = TfidfVectorizer()
    # data = cv.fit_transform(["I love you", "I like you"])
    # data = cv.fit_transform(["人生 苦短 我 喜欢 Python", "人生 漫长 我 讨厌 Python"])
    data = cv.fit_transform([' '.join(jieba.cut("人生苦短，我喜欢Python")), ' '.join(jieba.cut("人生漫长，我讨厌Python"))])
    print(data)
    print(cv.get_feature_names())
    print(data.toarray())

# 归一化
def normalize_data():
    """
    数据：
        30 10 20
        70 30 50
        110 50 35
    公式：
              x - min
        x' = —————————
             max - min

        x" = x'(mx - mi) + mi
        (default mx = 1, mi = 0)
    """
    mms = MinMaxScaler(feature_range=(2, 3))
    data = mms.fit_transform([[30, 10, 20], [70, 30, 50], [110, 50, 35]])
    print(data)

# 标准化
def standard_data():
    """
    公式：
        方差:
              (x1 - avg)^2 + (x2 - avg)^2 + ...
        var = —————————————————————————————————
                            n
        标准差：
             ___
        a = √var
             x - avg
        x' = ————————
                a
        avg为平均值,x'为最终结果
    """
    ss = StandardScaler()
    data = ss.fit_transform([[30, 10, 20], [70, 30, 50], [110, 50, 35]])
    print(data.toarray())

# 缺失数据，补充
def imputer_data():
    # 平均值补充
    ss = Imputer(missing_values="NaN", strategy="mean", axis=0)
    data = ss.fit_transform([[30, numpy.NAN, 20], [70, 30, 50], [110, 50, numpy.NAN]])
    print(data)

# 降维
def variance_data():
    # delete low variance
    # 过滤式
    # 对于方差减小的特征数据，进行特征删除，保留特征差异大的数据
    variance = VarianceThreshold(threshold=0.0)
    data = variance.fit_transform([[0, 10, 20], [0, 30, 50], [0, 50, 45]])
    print(data)

# 降维的一种处理方式，降低数据原始复杂度
def pca_data():
    # n_components（数据保留率一般90%~95%）
    pca = PCA(n_components=0.9)
    data = pca.fit_transform([[30, 10, 20], [70, 30, 50], [110, 50, 45]])
    print(data)

def iris_dataset():
    # 获取数据集
    li = load_iris()
    # print(li.data)
    # print(li.target)
    # print(li.DESCR)
    # 获取训练集合测试集结果
    x_train, x_test, y_train, y_test = train_test_split(li.data, li.target, test_size=0.25)
    print(x_train, y_train)
    print(x_test, y_test)

def boston_dataset():
    lb = load_boston()
    print(lb.data)
    print(lb.target)
    print(lb.DESCR)


"""
    监督学习：
        特征值+目标值：（有标准答案）
            分类：
                离散型（有具体的分类标准）
            回归：
                连续型（具体的预测值，不确定）
    无监督学习
        特征值：（无标准答案）
            聚类
    
    原始数据-->数据处理（合并，缺失）-->特征工程-->算法-->评估    
"""


if __name__ == '__main__':
    dict_data()
    # count_data()
    # tf_idf_data()
    # normalize_data()
    # standard_data()
    # imputer_data()
    # variance_data()
    # pca_data()
    # iris_dataset()
    # boston_dataset()