# !/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas
from sklearn.decomposition import PCA

# 降维
def dimensionality_reduction():
    # 读取数据
    orders = pandas.read_csv("market/orders.csv")
    prior = pandas.read_csv("market/order_products__prior.csv")
    products = pandas.read_csv("market/products.csv")
    aisles = pandas.read_csv("market/aisles.csv")
    # 合并数据
    _msg = pandas.merge(orders, prior, on=["order_id", "order_id"])
    _msg = pandas.merge(_msg, products, on=["product_id", "product_id"])
    merge_data = pandas.merge(_msg, aisles, on=["aisle_id", "aisle_id"])
    # 交叉表(特殊分组)
    # （用户ID， 类别）
    cross = pandas.crosstab(merge_data["user_id"], merge_data["aisle"])
    print(cross.shape)
    # 降维
    pca = PCA(n_components=0.9)
    pca.fit()
    data = pca.fit_transform(cross)
    print(data.shape)

if __name__ == '__main__':
    dimensionality_reduction()