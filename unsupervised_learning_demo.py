# !/usr/bin/python
# -*- coding: UTF-8 -*-
from matplotlib import pyplot
import pandas
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# 聚类
def k_means():
    """
        k值：
            分类个数，一般是知道分类个数的，如果不知道，进行超参数设置
        算法实现过程：
            1）随机在数据中抽取K个样本，当做K个类别的中心点
            2）计算其余的点到这K个点的距离，每一个样本有K个距离值，从中选出最近的一个距离点作为自己的标记
                这样就形成了K个族群
            3）计算着K个族群的平均值，把这K个平均值，与之前的K个中心点进行比较。
                如果相同：结束聚类
                如果不同：把K个平均值作为新的中心点，进行计算
        优点：
            采用迭代式算法，直观易懂并且非常实用
        缺点：
            容易收敛到局部最优解（多次聚类）
        注意：聚类一般是在做分类之前
    """
    # 1、原始数据
    orders = pandas.read_csv("market/orders.csv")
    prior = pandas.read_csv("market/order_products__prior.csv")
    products = pandas.read_csv("market/products.csv")
    aisles = pandas.read_csv("market/aisles.csv")

    # 2、数据处理
    # 合并数据
    _msg = pandas.merge(orders, prior, on=["order_id", "order_id"])
    _msg = pandas.merge(_msg, products, on=["product_id", "product_id"])
    merge_data = pandas.merge(_msg, aisles, on=["aisle_id", "aisle_id"])
    # 交叉表(特殊分组)
    # （用户ID， 类别）
    cross = pandas.crosstab(merge_data["user_id"], merge_data["aisle"])
    print(cross.shape)

    # 3、特征工程
    # 降维
    pca = PCA(n_components=0.9)
    data = pca.fit_transform(cross)
    print(data.shape)

    # 4、算法
    """
        n_clusters:开始均值的中心数量
    """
    km = KMeans(n_clusters=4)
    #减少数据量
    data = data[1:1000]
    # 训练
    km.fit(data)
    # 预测结果
    predict = km.predict(data)
    print("预测值：", predict)

    # 5、评估
    """
        轮廓系数：
                    bi - ai
            sci = ———————————
                  max(bi, ai)
        注：对于每个点i为已聚类数据中的样本，bi为i到其他族群的所有样本的距离
        最小值，ai为i到本族群的距离平均值
        最终算出所有的样本的轮廓系数平均值
        sci范围：[-1, 1],越靠近1越好
    """
    print("预测效果：", silhouette_score(data, predict))

    # 6、图形展示
    pyplot.figure(figsize=(10, 10))
    colors = ["red", "blue", "orange", "yellow"]
    color = [colors[i] for i in predict]
    pyplot.scatter(data[:, 1], data[:, 20], color=color)
    pyplot.xlabel("1")
    pyplot.ylabel("20")
    pyplot.show()

if __name__ == '__main__':
    k_means()

